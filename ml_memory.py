# ml_memory.py  — semantic memory (MiniLM) with SQLite storage
# Works on CPU. No FAISS needed. Uses HuggingFace transformers.

from __future__ import annotations
import os, json, sqlite3, time
from typing import List, Tuple, Dict
import numpy as np

# We use transformers directly so you don't need the sentence-transformers package
from transformers import AutoTokenizer, AutoModel
import torch

def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = torch.sum(masked, 1)
    counts = torch.clamp(mask.sum(1), min=1e-9)
    return summed / counts

class MemoryIndex:
    """
    Stores text rows in SQLite (table mem_docs) and keeps a parallel embedding
    matrix on disk (/data/memory/embeddings.npy + ids.json).
    API:
      - init(base_dir)            -> sets paths and prepares SQLite
      - start(model_name=...)     -> loads MiniLM model
      - add(id, text, coin, ts)   -> upsert a row + embedding
      - add_or_skip(id, text, ts, coins) -> helper for headlines
      - search(query, k=5)        -> list of {'id','ts','coin','text','score'}
      - rebuild()                 -> recompute embeddings from DB (rarely needed)
    """
    def __init__(self, base_dir: str):
        self.base_dir = os.path.join(base_dir, "memory")
        os.makedirs(self.base_dir, exist_ok=True)
        self.db_path = os.path.join(base_dir, "crypto.db")
        self.ids_path = os.path.join(self.base_dir, "ids.json")
        self.emb_path = os.path.join(self.base_dir, "embeddings.npy")

        self.tokenizer = None
        self.model = None
        self.dim = 384  # MiniLM-L6-v2 output size
        self.ids: List[str] = []
        self.emb: np.ndarray | None = None

        self.init()

    # -------- storage

    def init(self):
        with sqlite3.connect(self.db_path) as db:
            db.execute("""
              CREATE TABLE IF NOT EXISTS mem_docs(
                id   TEXT PRIMARY KEY,
                ts   TEXT,
                coin TEXT,
                text TEXT
              )""")
        # load cache files if they exist
        if os.path.exists(self.ids_path) and os.path.exists(self.emb_path):
            try:
                with open(self.ids_path, "r", encoding="utf-8") as f:
                    self.ids = json.load(f)
                self.emb = np.load(self.emb_path)
            except Exception:
                self.ids, self.emb = [], None

    # -------- model

    def start(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=False)
        self.model.eval()

    # -------- helpers

    def _encode(self, texts: List[str]) -> np.ndarray:
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("MemoryIndex model not started. Call start().")
        with torch.no_grad():
            batch = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=256)
            out = self.model(**batch)
            pooled = _mean_pool(out.last_hidden_state, batch["attention_mask"])
            emb = torch.nn.functional.normalize(pooled, p=2, dim=1)
            return emb.cpu().numpy().astype("float32")

    def _save_cache(self):
        if self.emb is None:
            return
        np.save(self.emb_path, self.emb)
        with open(self.ids_path, "w", encoding="utf-8") as f:
            json.dump(self.ids, f)

    # -------- public API

    def add(self, id: str, text: str, coin: str = "", ts: str | None = None):
        if not text:
            return
        # upsert in DB
        with sqlite3.connect(self.db_path) as db:
            db.execute("INSERT OR REPLACE INTO mem_docs(id, ts, coin, text) VALUES(?,?,?,?)",
                       (id, ts or "", coin or "", text))

        # append embedding to cache (recompute one vector)
        vec = self._encode([text])[0]  # (384,)
        if self.emb is None or len(self.ids) == 0:
            self.ids = [id]
            self.emb = vec.reshape(1, -1)
        else:
            # replace if exists, else append
            try:
                idx = self.ids.index(id)
                self.emb[idx] = vec
            except ValueError:
                self.ids.append(id)
                self.emb = np.vstack([self.emb, vec])
        self._save_cache()

    def add_or_skip(self, id: str, text: str, ts: str, coins: list[str] | None = None):
        # skip near-dup by id (ui already hashes title+url)
        if id in self.ids:
            return
        first_coin = (coins or [""])[0] if coins else ""
        self.add(id=id, text=text, coin=first_coin, ts=ts)

    def search(self, query: str, k: int = 5) -> List[Dict]:
        if not query:
            return []
        if self.emb is None or len(self.ids) == 0:
            return []
        qv = self._encode([query])[0]  # (384,)
        # cosine: since emb rows are L2-normalized, dot product == cosine
        scores = self.emb @ qv
        idx = np.argsort(-scores)[: max(1, k)]
        out = []
        with sqlite3.connect(self.db_path) as db:
            for i in idx:
                _id = self.ids[int(i)]
                s = float(scores[int(i)])
                cur = db.execute("SELECT id, ts, coin, text FROM mem_docs WHERE id=?", (_id,))
                row = cur.fetchone()
                if row:
                    out.append({
                        "id": row[0], "ts": row[1], "coin": row[2],
                        "text": row[3], "score": round(s, 3)
                    })
        return out

    def rebuild(self):
        """Recompute embeddings for all docs (only if you really need)."""
        with sqlite3.connect(self.db_path) as db:
            rows = db.execute("SELECT id, text FROM mem_docs ORDER BY ts DESC LIMIT 5000").fetchall()
        if not rows:
            self.ids, self.emb = [], None
            self._save_cache()
            return
        ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]
        EMB = []
        B = 64
        for i in range(0, len(texts), B):
            EMB.append(self._encode(texts[i:i+B]))
        self.ids = ids
        self.emb = np.vstack(EMB)
        self._save_cache()
