# -*- coding: utf-8 -*-
# =============================================================================
#  Crypto Intel — Phase 2 (Full App)
# =============================================================================
#  Features:
#   - FastAPI single-page UI (alerts, headlines, predictions, ticker)
#   - RSS -> news -> per-coin sentiment (FinBERT optional, lexicon fallback)
#   - Alerts (simple heuristic), metrics store (/debug/metrics)
#   - Binance prices ticker loop
#   - Daily trainer (Ridge) per-coin, /predictions with ML if available
#   - Phase 2: Semantic memory (MiniLM via ml_memory) + /memory/search
#
#  Reliability / Safety:
#   - MemoryIndex is optional; app runs without it
#   - /memory/search uses positional (q, k) signature
#   - All memory adds use add_or_skip(id, text, ts, coins)
#   - Defensive guards for empty/NaN sentiment windows
#   - Consistent indentation in all try/except blocks
#
#  Persisted Data (inside container, bind-mounted from host):
#     /data/crypto.db         -> SQLite database
#     /data/models            -> FinBERT cache + trained Ridge .pkl files
#     /data/memory            -> Semantic memory index files
#     /data/predictions.json  -> (optional file used by UI)
#
#  Endpoints:
#     GET /                      -> UI
#     GET /prices                -> Ticker cache
#     GET /predictions           -> Predictions per coin (+ metrics log)
#     GET /news                  -> Latest headlines from DB
#     GET /debug/metrics         -> Recent metrics from DB
#     GET /memory/search?q=...   -> Semantic search (if memory is ready)
#
#  Notes:
#     - Sklearn may warn "X does not have valid feature names" — harmless.
#     - Transformers may warn about return_all_scores — harmless.
#     - If no internet or FinBERT unavailable, falls back to lexicon.
#
# =============================================================================

from __future__ import annotations

# ------------------------------
# Imports
# ------------------------------
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from typing import List, Dict, Tuple, Any, Optional
import feedparser
import threading
import time
import os
import json
import sqlite3
import httpx
from hashlib import sha256
from datetime import datetime, timezone, timedelta
from dateutil import parser as dtparser
import pickle
import math
from pathlib import Path
VERSION = "1.1.0"
MODEL_VERSION = "2.1.0"
STARTED_AT = datetime.now(timezone.utc).isoformat()

# =============================================================================
# Phase-2: Semantic Memory (optional)
# =============================================================================
try:
    # Lightweight wrapper we added earlier (FAISS + MiniLM)
    from ml_memory import MemoryIndex
except Exception:
    MemoryIndex = None  # app runs without it

# single global index (created on startup if available)
MEM: Optional["MemoryIndex"] = None


# =============================================================================
# Paths & Constants
# =============================================================================
DB_DIR = "/data"
os.makedirs(DB_DIR, exist_ok=True)

DB_PATH = os.path.join(DB_DIR, "crypto.db")
PRED_PATH = os.path.join(DB_DIR, "predictions.json")

MODEL_DIR = os.path.join(DB_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MEM_DIR = os.path.join(DB_DIR, "memory")
os.makedirs(MEM_DIR, exist_ok=True)

TEMPLATE_PATH = Path(__file__).resolve().parent / "assets" / "dashboard_template.html"

MEMORY_MODEL = os.getenv("MEMORY_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MEMORY_LOCAL_ONLY = os.getenv("MEMORY_LOCAL_ONLY", "1") == "1"
MEMORY_ALLOW_DOWNLOAD = os.getenv("MEMORY_ALLOW_DOWNLOAD", "0") == "1"

BINANCE = "https://api.binance.com"

TICKER_SYMBOLS: List[str] = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "MATICUSDT", "DOTUSDT", "AVAXUSDT",
    "LINKUSDT", "TONUSDT",
]

COIN_META = {
    "BTC": {"icon": "₿", "color": "#F7931A"},
    "ETH": {"icon": "Ξ", "color": "#627EEA"},
    "SOL": {"icon": "S", "color": "#14F195"},
    "BNB": {"icon": "B", "color": "#F3BA2F"},
    "XRP": {"icon": "X", "color": "#23292F"},
    "ADA": {"icon": "A", "color": "#0033AD"},
    "DOGE": {"icon": "Ð", "color": "#C2A633"},
    "MATIC": {"icon": "M", "color": "#8247E5"},
    "DOT": {"icon": "P", "color": "#E6007A"},
    "AVAX": {"icon": "A", "color": "#E84142"},
    "LINK": {"icon": "L", "color": "#2A5ADA"},
    "TON": {"icon": "T", "color": "#0098EA"},
}

FEEDS = [
    "https://cointelegraph.com/rss",
    "https://news.bitcoin.com/feed/",
]

SOURCE_WEIGHTS = {
    "cointelegraph.com": 0.64,
    "news.bitcoin.com": 0.58,
}

# shared price cache for ticker bar
PRICES: Dict[str, Dict[str, float]] = {}
EVENT_HORIZONS = [1, 4, 24]


# =============================================================================
# DB bootstrap / helpers
# =============================================================================
def db_init() -> None:
    """Create required tables if they don't exist."""
    with sqlite3.connect(DB_PATH) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS news(
            id TEXT PRIMARY KEY,
            title TEXT,
            url TEXT,
            ts TEXT
        )""")

        db.execute("""CREATE TABLE IF NOT EXISTS alerts(
            id TEXT PRIMARY KEY,
            title TEXT,
            url TEXT,
            coin TEXT,
            score REAL,
            confidence TEXT,
            ts TEXT,
            reasons TEXT
        )""")
        db.execute("CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(ts)")

        # per-article sentiment
        db.execute("""CREATE TABLE IF NOT EXISTS sentiments(
            nid TEXT,
            coin TEXT,
            ts   TEXT,
            score REAL,
            source TEXT,
            PRIMARY KEY (nid, coin)
        )""")
        db.execute("CREATE INDEX IF NOT EXISTS idx_sentiments_coin_ts ON sentiments(coin, ts)")

        # trained model metadata
        db.execute("""CREATE TABLE IF NOT EXISTS models(
            coin TEXT PRIMARY KEY,
            trained_at TEXT,
            horizon_hours INTEGER,
            n_samples INTEGER,
            r2 REAL,
            path TEXT
        )""")

        # metrics
        db.execute("""CREATE TABLE IF NOT EXISTS ml_metrics(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            kind TEXT,
            coin TEXT,
            payload TEXT
        )""")

        # event-centric AI tables
        db.execute("""CREATE TABLE IF NOT EXISTS events(
            id TEXT PRIMARY KEY,
            nid TEXT,
            coin TEXT,
            ts TEXT,
            title TEXT,
            source_url TEXT,
            sentiment REAL,
            novelty REAL,
            features TEXT
        )""")
        db.execute("CREATE INDEX IF NOT EXISTS idx_events_coin_ts ON events(coin, ts)")

        db.execute("""CREATE TABLE IF NOT EXISTS event_predictions(
            event_id TEXT,
            horizon_h INTEGER,
            ts TEXT,
            model_version TEXT,
            direction TEXT,
            expected_return REAL,
            probability_up REAL,
            confidence REAL,
            reasons TEXT,
            PRIMARY KEY(event_id, horizon_h)
        )""")
        db.execute("CREATE INDEX IF NOT EXISTS idx_event_predictions_ts ON event_predictions(ts)")

        db.execute("""CREATE TABLE IF NOT EXISTS event_outcomes(
            event_id TEXT,
            horizon_h INTEGER,
            resolved_at TEXT,
            entry_price REAL,
            exit_price REAL,
            realized_return REAL,
            hit INTEGER,
            PRIMARY KEY(event_id, horizon_h)
        )""")

        db.execute("""CREATE TABLE IF NOT EXISTS model_eval(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            coin TEXT,
            model_version TEXT,
            horizon_h INTEGER,
            n_test INTEGER,
            direction_acc REAL,
            mae REAL,
            baseline_acc REAL
        )""")

        # horizon-specific model metadata (phase upgrade)
        db.execute("""CREATE TABLE IF NOT EXISTS models_horizon(
            coin TEXT,
            horizon_hours INTEGER,
            trained_at TEXT,
            n_samples INTEGER,
            quality_score REAL,
            path TEXT,
            PRIMARY KEY (coin, horizon_hours)
        )""")


def save_metric(kind: str, ts: str, coin: str, payload: dict) -> None:
    """Insert one metric row; errors are swallowed."""
    try:
        with sqlite3.connect(DB_PATH) as db:
            db.execute(
                "INSERT INTO ml_metrics(ts, kind, coin, payload) VALUES(?,?,?,?)",
                (ts, kind, coin, json.dumps(payload)),
            )
    except Exception as e:
        print("save_metric error:", e)


def db_add_news(item: dict) -> None:
    with sqlite3.connect(DB_PATH) as db:
        db.execute(
            "INSERT OR REPLACE INTO news VALUES(?,?,?,?)",
            (item["id"], item["title"], item["url"], item["ts"]),
        )


def db_get_news(limit: int = 50) -> List[dict]:
    with sqlite3.connect(DB_PATH) as db:
        cur = db.execute("SELECT * FROM news ORDER BY ts DESC LIMIT ?", (limit,))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def db_add_alert(item: dict) -> None:
    with sqlite3.connect(DB_PATH) as db:
        db.execute(
            "INSERT OR REPLACE INTO alerts VALUES(?,?,?,?,?,?,?,?)",
            (
                item["id"],
                item["title"],
                item["url"],
                item["coin"],
                item["score"],
                item["confidence"],
                item["ts"],
                item["reasons"],
            ),
        )


def db_get_alerts_between(start_iso: str, end_iso: str, limit: int = 200) -> List[dict]:
    with sqlite3.connect(DB_PATH) as db:
        cur = db.execute(
            "SELECT * FROM alerts WHERE ts>=? AND ts<? ORDER BY ts DESC LIMIT ?",
            (start_iso, end_iso, limit),
        )
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def db_upsert_event(item: dict, db: Optional[sqlite3.Connection] = None) -> None:
    owns_connection = db is None
    if db is None:
        db = sqlite3.connect(DB_PATH)

    try:
        db.execute(
            """INSERT OR REPLACE INTO events
               (id, nid, coin, ts, title, source_url, sentiment, novelty, features)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                item["id"],
                item["nid"],
                item["coin"],
                item["ts"],
                item["title"],
                item["source_url"],
                float(item["sentiment"]),
                float(item["novelty"]),
                json.dumps(item.get("features", {})),
            ),
        )
        if owns_connection:
            db.commit()
    finally:
        if owns_connection:
            db.close()


def db_upsert_event_prediction(item: dict, db: Optional[sqlite3.Connection] = None) -> None:
    owns_connection = db is None
    if db is None:
        db = sqlite3.connect(DB_PATH)

    try:
        db.execute(
            """INSERT OR REPLACE INTO event_predictions
               (event_id, horizon_h, ts, model_version, direction, expected_return,
                probability_up, confidence, reasons)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                item["event_id"],
                int(item["horizon_h"]),
                item["ts"],
                item["model_version"],
                item["direction"],
                float(item["expected_return"]),
                float(item["probability_up"]),
                float(item["confidence"]),
                json.dumps(item.get("reasons", [])),
            ),
        )
        if owns_connection:
            db.commit()
    finally:
        if owns_connection:
            db.close()


def db_insert_outcome(item: dict) -> None:
    with sqlite3.connect(DB_PATH) as db:
        db.execute(
            """INSERT OR REPLACE INTO event_outcomes
               (event_id, horizon_h, resolved_at, entry_price, exit_price, realized_return, hit)
               VALUES(?,?,?,?,?,?,?)""",
            (
                item["event_id"],
                int(item["horizon_h"]),
                item["resolved_at"],
                float(item["entry_price"]),
                float(item["exit_price"]),
                float(item["realized_return"]),
                int(item["hit"]),
            ),
        )


# =============================================================================
# Utilities
# =============================================================================
def normalize_id(title: str, url: str) -> str:
    return sha256((title + "|" + url).encode("utf-8")).hexdigest()[:24]


COIN_MATCHES = {
    "bitcoin": "BTC",
    "btc": "BTC",
    "ethereum": "ETH",
    "eth": "ETH",
    "solana": "SOL",
    "sol": "SOL",
    "binance coin": "BNB",
    "bnb": "BNB",
    "ripple": "XRP",
    "xrp": "XRP",
    "cardano": "ADA",
    "ada": "ADA",
    "dogecoin": "DOGE",
    "doge": "DOGE",
    "polygon": "MATIC",
    "matic": "MATIC",
    "polkadot": "DOT",
    "dot": "DOT",
    "avalanche": "AVAX",
    "avax": "AVAX",
    "chainlink": "LINK",
    "link": "LINK",
    "ton": "TON",
}


def coins_in_title(title: str) -> List[str]:
    t = f" {title.lower()} "
    found = set()
    for k, c in COIN_MATCHES.items():
        if f" {k} " in t or t.endswith(f" {k}") or t.startswith(f"{k} "):
            found.add(c)
    for sym in ("BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "MATIC", "DOT", "AVAX", "LINK", "TON"):
        if f" {sym.lower()} " in t or sym in title:
            found.add(sym)
    return sorted(found)


def window_ends() -> dict:
    now = datetime.now(timezone.utc)
    return {
        "day": (now - timedelta(days=1)).isoformat(),
        "week": (now - timedelta(days=7)).isoformat(),
        "month": (now - timedelta(days=30)).isoformat(),
        "year": (now - timedelta(days=365)).isoformat(),
        "now": now.isoformat(),
    }


def load_predictions() -> dict:
    try:
        with open(PRED_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"asof": None, "window_hours": 48, "coins": []}


# =============================================================================
# FinBERT (optional)
# =============================================================================
_FINBERT = None


def finbert():
    """Lazy-load FinBERT, or return None if unavailable."""
    global _FINBERT
    if _FINBERT is not None:
        return _FINBERT
    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TextClassificationPipeline,
        )

        tok = AutoTokenizer.from_pretrained("ProsusAI/finbert", cache_dir=MODEL_DIR)
        mdl = AutoModelForSequenceClassification.from_pretrained(
            "ProsusAI/finbert", cache_dir=MODEL_DIR
        )
        _FINBERT = TextClassificationPipeline(
            model=mdl,
            tokenizer=tok,
            return_all_scores=True,
            truncation=True,
        )
        print("FinBERT loaded.")
        return _FINBERT
    except Exception as e:
        print("FinBERT unavailable, using lexicon fallback. Reason:", e)
        return None


def weak_fallback_sentiment(title: str) -> float:
    """Tiny lexicon fallback in [-1..+1]."""
    POS = (
        "surge",
        "rally",
        "bull",
        "boom",
        "gain",
        "adoption",
        "etf",
        "approval",
        "partnership",
        "upgrade",
        "support",
        "breakout",
    )
    NEG = (
        "hack",
        "ban",
        "crash",
        "bear",
        "lawsuit",
        "exploit",
        "halt",
        "down",
        "sell-off",
        "delist",
        "sanction",
    )
    t = title.lower()
    s = 0.0
    s += sum(w in t for w in POS) * 0.4
    s -= sum(w in t for w in NEG) * 0.4
    return max(-1.0, min(1.0, s))


def signed_sentiment(title: str) -> float:
    """Prefer FinBERT; fall back to lexicon. Returns [-1..+1]."""
    fb = finbert()
    if fb:
        try:
            out = fb(title[:512])[0]
            scores = {d["label"].lower(): d["score"] for d in out}
            return float(scores.get("positive", 0.0) - scores.get("negative", 0.0))
        except Exception:
            pass
    return weak_fallback_sentiment(title)


def sentiment_keywords(title: str) -> Dict[str, float]:
    t = title.lower()
    topic_weights = {
        "etf": 0.25,
        "approval": 0.20,
        "partnership": 0.15,
        "adoption": 0.15,
        "upgrade": 0.10,
        "hack": -0.30,
        "lawsuit": -0.25,
        "ban": -0.20,
        "exploit": -0.25,
        "liquidation": -0.15,
    }
    out = {"topic_bias": 0.0, "keyword_hits": 0.0}
    for k, w in topic_weights.items():
        if k in t:
            out["topic_bias"] += w
            out["keyword_hits"] += 1.0
    return out


def logistic(x: float) -> float:
    x = max(-10.0, min(10.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def parse_iso(ts: str) -> datetime:
    return dtparser.parse(ts).astimezone(timezone.utc)


def source_key(url: str) -> str:
    try:
        host = httpx.URL(url).host or ""
    except Exception:
        host = ""
    return host.lower().strip()


def source_reliability(url: str) -> float:
    host = source_key(url)
    if not host:
        return 0.50
    for k, v in SOURCE_WEIGHTS.items():
        if host.endswith(k):
            return float(v)
    return 0.52


def estimate_novelty(title: str) -> float:
    """Return novelty in [0..1], where higher means more novel headline."""
    if MEM is not None:
        try:
            hits = MEM.search(title, 1)
            if hits:
                top = hits[0]
                sim = float(top.get("score", 0.0))
                return max(0.05, min(1.0, 1.0 - sim))
        except Exception:
            pass

    try:
        with sqlite3.connect(DB_PATH) as db:
            row = db.execute(
                "SELECT title FROM news ORDER BY ts DESC LIMIT 200"
            ).fetchall()
        t0 = set(title.lower().split())
        best = 0.0
        for (prev,) in row:
            t1 = set((prev or "").lower().split())
            if not t0 or not t1:
                continue
            jac = len(t0 & t1) / max(1, len(t0 | t1))
            if jac > best:
                best = jac
        return max(0.05, min(1.0, 1.0 - best))
    except Exception:
        return 0.50


def init_memory() -> None:
    """Best-effort semantic memory init; never block app startup on network downloads."""
    global MEM
    if MemoryIndex is None:
        print("[Memory] package not available; skipping.")
        return

    try:
        # IMPORTANT: base_dir must be the root data folder (/data), not /data/memory.
        # MemoryIndex derives both /data/crypto.db and /data/memory/* paths from this.
        MEM = MemoryIndex(base_dir=DB_DIR)
        if MEMORY_LOCAL_ONLY:
            MEM.start(model_name=MEMORY_MODEL, local_files_only=True)
            print("[Memory] semantic index ready (local model cache).")
            return
        if MEMORY_ALLOW_DOWNLOAD:
            MEM.start(model_name=MEMORY_MODEL, local_files_only=False)
            print("[Memory] semantic index ready (download enabled).")
            return

        # Default: safe startup policy. Try local cache only and keep app running if absent.
        MEM.start(model_name=MEMORY_MODEL, local_files_only=True)
        print("[Memory] semantic index ready (default local-only mode).")
    except Exception as e:
        MEM = None
        print("[Memory] disabled:", e)


def current_price(symbol: str) -> Optional[float]:
    snap = PRICES.get(symbol)
    if snap:
        return float(snap.get("last", 0.0)) or None
    try:
        with httpx.Client(timeout=6) as c:
            r = c.get(f"{BINANCE}/api/v3/ticker/price", params={"symbol": symbol})
            r.raise_for_status()
            return float(r.json()["price"])
    except Exception:
        return None


def event_price_at_or_near(symbol: str, target_ts: datetime) -> Optional[float]:
    """Get close price near target_ts using 1h candles."""
    try:
        ms = int(target_ts.timestamp() * 1000)
        start = ms - 2 * 3600 * 1000
        end = ms + 2 * 3600 * 1000
        with httpx.Client(timeout=8) as c:
            r = c.get(
                f"{BINANCE}/api/v3/klines",
                params={"symbol": symbol, "interval": "1h", "startTime": start, "endTime": end},
            )
            r.raise_for_status()
            data = r.json()
        if not data:
            return current_price(symbol)
        best = None
        best_dt = 10**18
        for k in data:
            ts_ms = int(k[0])
            delta = abs(ts_ms - ms)
            if delta < best_dt:
                best_dt = delta
                best = float(k[4])
        return best
    except Exception:
        return None


def build_event_prediction(coin: str, sentiment: float, title: str, ts: str, source_url: str) -> dict:
    f = sentiment_keywords(title)
    now_price = current_price(f"{coin}USDT") or 0.0
    src_rel = source_reliability(source_url)
    novelty = estimate_novelty(title)
    base = (
        0.50 * sentiment
        + 0.25 * f["topic_bias"]
        + 0.15 * (src_rel - 0.5)
        + 0.10 * (novelty - 0.5)
    )
    expected = max(-0.08, min(0.08, base * 0.04))
    prob_up = logistic(base * 2.4)
    confidence = min(0.98, max(0.05, abs(base) + 0.15 * src_rel))
    reasons = [
        f"sentiment={sentiment:+.3f}",
        f"topic_bias={f['topic_bias']:+.3f}",
        f"keyword_hits={int(f['keyword_hits'])}",
        f"source_rel={src_rel:.2f}",
        f"novelty={novelty:.2f}",
        f"price_snapshot={now_price:.2f}",
    ]
    return {
        "coin": coin,
        "ts": ts,
        "expected_return": expected,
        "probability_up": prob_up,
        "confidence": confidence,
        "direction": "up" if expected >= 0 else "down",
        "reasons": reasons,
        "feature_map": {
            "sentiment": sentiment,
            "topic_bias": f["topic_bias"],
            "keyword_hits": f["keyword_hits"],
            "source_reliability": src_rel,
            "novelty": novelty,
            "price_snapshot": now_price,
        },
    }


def predict_horizon_expected(coin: str, horizon_h: int, feature_map: Dict[str, float]) -> Optional[float]:
    try:
        with sqlite3.connect(DB_PATH) as db:
            row = db.execute(
                "SELECT path FROM models_horizon WHERE coin=? AND horizon_hours=?",
                (coin, int(horizon_h)),
            ).fetchone()
        if not row:
            return None
        with open(row[0], "rb") as f:
            obj = pickle.load(f)
        mdl = obj.get("model") if isinstance(obj, dict) else obj
        feats = obj.get("features", []) if isinstance(obj, dict) else []
        if mdl is None:
            return None
        if not feats:
            feats = ["ema6", "ema24", "cnt", "ret_1h", "ret_6h", "ret_24h", "vol_24h", "topic_bias", "keyword_hits", "source_reliability", "novelty"]
        x = [[float(feature_map.get(name, 0.0)) for name in feats]]
        yhat = float(mdl.predict(x)[0])
        if math.isfinite(yhat):
            return max(-0.10, min(0.10, yhat))
    except Exception:
        return None
    return None


# =============================================================================
# Binance helpers
# =============================================================================
def klines_close_prices(symbol: str, days: int = 30, interval: str = "1h") -> List[Tuple[str, float]]:
    """Fetch (iso_ts, close) tuples from Binance."""
    end = int(time.time() * 1000)
    start = end - days * 24 * 60 * 60 * 1000
    url = f"{BINANCE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "startTime": start, "endTime": end}
    with httpx.Client(timeout=8) as c:
        r = c.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    out = []
    for k in data:
        ts = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc).isoformat()
        close = float(k[4])
        out.append((ts, close))
    return out


# =============================================================================
# Long-running loops
# =============================================================================
def fetch_rss_once() -> List[dict]:
    """Fetch one pass of RSS headlines."""
    items = []
    for url in FEEDS:
        feed = feedparser.parse(url)
        for e in feed.entries[:40]:
            title = e.get("title", "").strip()
            link = e.get("link", "")
            published = e.get("published", "") or e.get("updated", "")
            try:
                ts = dtparser.parse(published).astimezone(timezone.utc)
            except Exception:
                ts = datetime.now(timezone.utc)
            items.append(
                {
                    "id": normalize_id(title, link),
                    "title": title,
                    "url": link,
                    "ts": ts.isoformat(),
                    "source": source_key(url),
                }
            )
    return items


def rss_loop():
    """Continuously fetch RSS, create event records, score sentiment, and feed memory."""
    while True:
        try:
            for n in fetch_rss_once():
                db_add_news(n)
                cs = coins_in_title(n["title"])
                if not cs:
                    continue

                sent = signed_sentiment(n["title"])
                with sqlite3.connect(DB_PATH) as db:
                    for coin in cs:
                        db.execute(
                            "INSERT OR REPLACE INTO sentiments(nid, coin, ts, score, source) VALUES(?,?,?,?,?)",
                            (n["id"], coin, n["ts"], float(sent), "finbert" if _FINBERT else "lexicon"),
                        )

                        event_id = normalize_id(f"{n['id']}:{coin}", n["ts"])
                        pred = build_event_prediction(coin, sent, n["title"], n["ts"], n["url"])
                        db_upsert_event(
                            {
                                "id": event_id,
                                "nid": n["id"],
                                "coin": coin,
                                "ts": n["ts"],
                                "title": n["title"],
                                "source_url": n["url"],
                                "sentiment": sent,
                                "novelty": pred["feature_map"].get("novelty", 0.5),
                                "features": pred["feature_map"],
                            },
                            db=db,
                        )
                        for h in EVENT_HORIZONS:
                            expected_h = predict_horizon_expected(
                                coin,
                                h,
                                {
                                    **pred["feature_map"],
                                    "ema6": sent,
                                    "ema24": sent,
                                    "cnt": 1.0,
                                    "ret_1h": 0.0,
                                    "ret_6h": 0.0,
                                    "ret_24h": 0.0,
                                    "vol_24h": 0.0,
                                },
                            )
                            expected_h = expected_h if expected_h is not None else pred["expected_return"] * (h / 24.0)
                            prob_up_h = logistic(expected_h * 35.0)
                            db_upsert_event_prediction(
                                {
                                    "event_id": event_id,
                                    "horizon_h": h,
                                    "ts": n["ts"],
                                    "model_version": MODEL_VERSION,
                                    "direction": "up" if expected_h >= 0 else "down",
                                    "expected_return": expected_h,
                                    "probability_up": prob_up_h,
                                    "confidence": pred["confidence"],
                                    "reasons": pred["reasons"] + [f"horizon={h}h"],
                                },
                                db=db,
                            )

                try:
                    if MEM is not None:
                        MEM.add_or_skip(n["id"], n["title"], n["ts"], cs)
                except Exception as me:
                    print("Memory add error (headline):", me)

        except Exception as e:
            print("RSS loop error:", e)

        time.sleep(300)


def alert_generation_loop() -> None:
    """Generate alerts from event predictions instead of random placeholders."""
    while True:
        try:
            with sqlite3.connect(DB_PATH) as db:
                rows = db.execute(
                    """
                    SELECT e.id, e.title, e.source_url, e.coin, e.ts, e.features,
                           p.expected_return, p.probability_up, p.confidence, p.reasons
                    FROM events e
                    JOIN event_predictions p ON p.event_id = e.id
                    WHERE p.horizon_h = 24
                    ORDER BY e.ts DESC
                    LIMIT 60
                    """
                ).fetchall()

            for eid, title, url, coin, ts, features_json, expected_ret, prob_up, conf, reasons_json in rows:
                feat = {}
                try:
                    feat = json.loads(features_json or "{}")
                except Exception:
                    feat = {}
                novelty = float(feat.get("novelty", 0.5))
                src_rel = float(feat.get("source_reliability", source_reliability(url)))
                conviction = max(prob_up, 1.0 - prob_up)
                score = float(expected_ret) * (6.0 + 4.0 * conf) * (0.6 + 0.4 * src_rel) * (0.7 + 0.6 * novelty)
                confidence_val = conf * conviction
                confidence = "High" if confidence_val >= 0.60 else "Med" if confidence_val >= 0.35 else "Low"
                reasons = []
                try:
                    reasons = json.loads(reasons_json)
                except Exception:
                    reasons = ["model-prediction"]
                reasons.extend([
                    f"source_rel={src_rel:.2f}",
                    f"novelty={novelty:.2f}",
                    f"p_up={prob_up:.2f}",
                ])
                db_add_alert(
                    {
                        "id": f"alert-{eid}",
                        "title": title,
                        "url": url,
                        "coin": coin,
                        "score": round(score, 3),
                        "confidence": confidence,
                        "ts": ts,
                        "reasons": "; ".join(reasons[:4]),
                    }
                )
        except Exception as e:
            print("Alert loop error:", e)
        time.sleep(90)


def outcome_resolution_loop() -> None:
    """Resolve event predictions by checking realized returns after target horizons."""
    while True:
        try:
            now = datetime.now(timezone.utc)
            with sqlite3.connect(DB_PATH) as db:
                rows = db.execute(
                    """
                    SELECT p.event_id, p.horizon_h, p.direction, e.coin, e.ts
                    FROM event_predictions p
                    JOIN events e ON e.id = p.event_id
                    LEFT JOIN event_outcomes o
                      ON o.event_id = p.event_id AND o.horizon_h = p.horizon_h
                    WHERE o.event_id IS NULL
                    ORDER BY e.ts ASC
                    LIMIT 200
                    """
                ).fetchall()

            for event_id, horizon_h, direction, coin, event_ts in rows:
                evt_dt = parse_iso(event_ts)
                target = evt_dt + timedelta(hours=int(horizon_h))
                if target > now:
                    continue
                symbol = f"{coin}USDT"
                entry = event_price_at_or_near(symbol, evt_dt)
                exit_ = event_price_at_or_near(symbol, target)
                if not entry or not exit_:
                    continue
                realized = (exit_ / entry) - 1.0
                hit = int((direction == "up" and realized >= 0) or (direction == "down" and realized < 0))
                db_insert_outcome(
                    {
                        "event_id": event_id,
                        "horizon_h": horizon_h,
                        "resolved_at": now.isoformat(),
                        "entry_price": entry,
                        "exit_price": exit_,
                        "realized_return": realized,
                        "hit": hit,
                    }
                )
                save_metric(
                    "event_outcome",
                    now.isoformat(),
                    coin,
                    {
                        "event_id": event_id,
                        "horizon_h": horizon_h,
                        "realized_return": realized,
                        "hit": hit,
                    },
                )
        except Exception as e:
            print("Outcome resolution loop error:", e)
        time.sleep(120)


def price_loop() -> None:
    """Fetch prices every ~12s and fill PRICES cache."""
    global PRICES
    while True:
        try:
            with httpx.Client(timeout=6) as c:
                symbols_param = json.dumps(TICKER_SYMBOLS, separators=(",", ":"))
                r = c.get(f"{BINANCE}/api/v3/ticker/24hr", params={"symbols": symbols_param})
                if r.status_code == 200:
                    data = r.json()
                else:
                    # fall back to individual calls if bulk call not allowed
                    data = []
                    for sym in TICKER_SYMBOLS:
                        rr = c.get(f"{BINANCE}/api/v3/ticker/24hr", params={"symbol": sym})
                        rr.raise_for_status()
                        data.append(rr.json())

                out = {}
                for d in data:
                    sym = d["symbol"]
                    out[sym] = {
                        "last": float(d["lastPrice"]),
                        "changePct": float(d["priceChangePercent"]),
                        "changeAbs": float(d["priceChange"]),
                    }
                PRICES = out
                print("Fetched Binance prices:", len(out), "coins")
        except Exception as e:
            print("Ticker loop error:", e)
        time.sleep(12)


def daily_trainer_loop() -> None:
    """
    Every ~6h: train per-coin Ridge models for 1h/4h/24h horizons,
    with walk-forward holdout metrics.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error
    import numpy as np
    import pandas as pd

    while True:
        try:
            print("Trainer: starting pass…")
            for full in TICKER_SYMBOLS:
                coin = full.replace("USDT", "")
                try:
                    kl = klines_close_prices(full, days=45, interval="1h")
                    if len(kl) < 96:
                        continue

                    dfp = pd.DataFrame(kl, columns=["ts", "close"]).set_index("ts")
                    dfp.index = pd.to_datetime(dfp.index)
                    dfp["ret_1h"] = dfp["close"].pct_change(1)
                    dfp["ret_6h"] = dfp["close"].pct_change(6)
                    dfp["ret_24h"] = dfp["close"].pct_change(24)
                    dfp["vol_24h"] = dfp["ret_1h"].rolling(24, min_periods=3).std().fillna(0.0)

                    since_iso = dfp.index.min().isoformat()
                    with sqlite3.connect(DB_PATH) as db:
                        srows = db.execute(
                            "SELECT ts, score FROM sentiments WHERE coin=? AND ts>=? ORDER BY ts ASC",
                            (coin, since_iso),
                        ).fetchall()
                        erows = db.execute(
                            "SELECT ts, features FROM events WHERE coin=? AND ts>=? ORDER BY ts ASC",
                            (coin, since_iso),
                        ).fetchall()

                    if not srows:
                        continue

                    dfs = pd.DataFrame(srows, columns=["ts", "score"]).set_index("ts")
                    dfs.index = pd.to_datetime(dfs.index)
                    dfs = dfs.resample("1h").mean().fillna(0.0)
                    dfs["ema6"] = dfs["score"].ewm(span=6, adjust=False).mean()
                    dfs["ema24"] = dfs["score"].ewm(span=24, adjust=False).mean()
                    dfs["cnt"] = (dfs["score"] != 0).astype(int).rolling(24, min_periods=1).sum()

                    if erows:
                        event_rows = []
                        for ets, fjson in erows:
                            feat = {}
                            try:
                                feat = json.loads(fjson or "{}")
                            except Exception:
                                feat = {}
                            event_rows.append(
                                {
                                    "ts": ets,
                                    "topic_bias": float(feat.get("topic_bias", 0.0)),
                                    "keyword_hits": float(feat.get("keyword_hits", 0.0)),
                                    "source_reliability": float(feat.get("source_reliability", 0.5)),
                                    "novelty": float(feat.get("novelty", 0.5)),
                                }
                            )
                        dfe = pd.DataFrame(event_rows).set_index("ts")
                        dfe.index = pd.to_datetime(dfe.index)
                        dfe = dfe.resample("1h").mean().fillna(0.0)
                    else:
                        dfe = pd.DataFrame(index=dfs.index)
                        dfe["topic_bias"] = 0.0
                        dfe["keyword_hits"] = 0.0
                        dfe["source_reliability"] = 0.5
                        dfe["novelty"] = 0.5

                    d = dfp.join(dfs[["ema6", "ema24", "cnt"]], how="left").join(
                        dfe[["topic_bias", "keyword_hits", "source_reliability", "novelty"]], how="left"
                    ).fillna(0.0)
                    feats = [
                        "ema6", "ema24", "cnt", "ret_1h", "ret_6h", "ret_24h", "vol_24h",
                        "topic_bias", "keyword_hits", "source_reliability", "novelty",
                    ]

                    for horizon_hours in EVENT_HORIZONS:
                        d[f"ret_next_{horizon_hours}h"] = (
                            d["close"].pct_change(periods=horizon_hours).shift(-horizon_hours)
                        )
                        y = d[f"ret_next_{horizon_hours}h"].dropna()
                        X = d.loc[y.index, feats]
                        if len(y) < 96:
                            continue

                        n = len(y)
                        split = int(n * 0.8)
                        Xtr, Xte = X.iloc[:split], X.iloc[split:]
                        ytr, yte = y.iloc[:split], y.iloc[split:]

                        mdl = Ridge(alpha=0.8).fit(Xtr, ytr)
                        yhat = mdl.predict(Xte)
                        mae = float(mean_absolute_error(yte, yhat))
                        acc = float(np.mean((yhat >= 0) == (yte.values >= 0)))
                        baseline = float(np.mean(yte.values >= 0))
                        score = max(-1.0, min(1.0, (acc - baseline) - mae))

                        art = {
                            "coin": coin,
                            "trained_at": datetime.now(timezone.utc).isoformat(),
                            "horizon_hours": int(horizon_hours),
                            "n_samples": int(n),
                            "quality_score": score,
                            "path": os.path.join(MODEL_DIR, f"{coin}_{horizon_hours}h.pkl"),
                        }
                        with open(art["path"], "wb") as f:
                            pickle.dump({"model": mdl, "features": feats, "version": MODEL_VERSION}, f)

                        with sqlite3.connect(DB_PATH) as db:
                            db.execute(
                                """INSERT OR REPLACE INTO models_horizon
                                   (coin, horizon_hours, trained_at, n_samples, quality_score, path)
                                   VALUES(?,?,?,?,?,?)""",
                                (
                                    art["coin"], art["horizon_hours"], art["trained_at"],
                                    art["n_samples"], art["quality_score"], art["path"],
                                ),
                            )
                            if horizon_hours == 24:
                                db.execute(
                                    "INSERT OR REPLACE INTO models VALUES(?,?,?,?,?,?)",
                                    (
                                        art["coin"], art["trained_at"], art["horizon_hours"],
                                        art["n_samples"], art["quality_score"], art["path"],
                                    ),
                                )
                            db.execute(
                                """INSERT INTO model_eval
                                   (ts, coin, model_version, horizon_h, n_test, direction_acc, mae, baseline_acc)
                                   VALUES(?,?,?,?,?,?,?,?)""",
                                (
                                    art["trained_at"], coin, MODEL_VERSION, horizon_hours,
                                    int(len(yte)), acc, mae, baseline,
                                ),
                            )
                        print(
                            f"Trainer: {coin} h={horizon_hours} acc={acc:.3f} "
                            f"mae={mae:.4f} n={n}"
                        )
                except Exception as e:
                    print(f"Trainer coin error {coin}:", e)
        except Exception as e:
            print("Trainer loop error:", e)

        time.sleep(6 * 60 * 60)
app = FastAPI(title="Crypto Intel")


@app.on_event("startup")
def on_start():
    global MEM

    db_init()

    # Start background loops
    threading.Thread(target=rss_loop, daemon=True).start()
    threading.Thread(target=alert_generation_loop, daemon=True).start()
    threading.Thread(target=price_loop, daemon=True).start()
    threading.Thread(target=outcome_resolution_loop, daemon=True).start()
    try:
        threading.Thread(target=daily_trainer_loop, daemon=True).start()
    except NameError:
        pass

    # Start semantic memory in background (non-blocking startup)
    threading.Thread(target=init_memory, daemon=True).start()


# =============================================================================
# API endpoints
# =============================================================================
@app.get("/prices")
def prices_api():
    return JSONResponse(PRICES)

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/version")
def version():
    # helpful when you hit the container to confirm what’s running
    return {"version": VERSION}

@app.get("/debug/metrics")
def debug_metrics(limit: int = 200):
    with sqlite3.connect(DB_PATH) as db:
        cur = db.execute(
            "SELECT ts, kind, coin, payload FROM ml_metrics ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = []
        for ts, kind, coin, payload in cur.fetchall():
            try:
                pl = json.loads(payload)
            except Exception:
                pl = {"raw": payload}
            rows.append({"ts": ts, "kind": kind, "coin": coin, "payload": pl})
    return rows


@app.get("/memory/search")
def memory_search(q: str, k: int = 5):
    """Semantic search over remembered headlines and prediction notes."""
    if MEM is None:
        return {"q": q, "hits": []}
    try:
        # IMPORTANT: our MemoryIndex.search signature is (query, k) — no top_k keyword
        hits = MEM.search(q, k)
        return {"q": q, "hits": hits}
    except Exception as e:
        return {"q": q, "error": str(e), "hits": []}






@app.get("/debug/runtime")
def debug_runtime():
    return {
        "app_version": VERSION,
        "model_version": MODEL_VERSION,
        "started_at": STARTED_AT,
        "now": datetime.now(timezone.utc).isoformat(),
    }

@app.get("/debug/model-quality")
def model_quality(limit: int = 120):
    with sqlite3.connect(DB_PATH) as db:
        rows = db.execute(
            """
            SELECT ts, coin, model_version, horizon_h, n_test, direction_acc, mae, baseline_acc
            FROM model_eval
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        {
            "ts": r[0],
            "coin": r[1],
            "model_version": r[2],
            "horizon_h": r[3],
            "n_test": r[4],
            "direction_acc": r[5],
            "mae": r[6],
            "baseline_acc": r[7],
        }
        for r in rows
    ]

@app.get("/news")
def news_api():
    return JSONResponse(db_get_news(100))


@app.get("/predictions")
def predictions_api(window_hours: int = 48):
    """
    Return per-coin prediction with probability, expected move, confidence,
    model metadata, and reason codes.
    """
    import numpy as np
    import pandas as pd

    now = datetime.now(timezone.utc)
    since = (now - timedelta(hours=window_hours)).isoformat()

    out = []
    with sqlite3.connect(DB_PATH) as db:
        for full in TICKER_SYMBOLS:
            coin = full.replace("USDT", "")
            rows = db.execute(
                "SELECT ts, score FROM sentiments WHERE coin=? AND ts>=? ORDER BY ts ASC",
                (coin, since),
            ).fetchall()
            erows = db.execute(
                "SELECT features FROM events WHERE coin=? AND ts>=? ORDER BY ts DESC LIMIT 60",
                (coin, since),
            ).fetchall()
            sample = len(rows)

            if rows:
                df = pd.DataFrame(rows, columns=["ts", "score"]).set_index("ts")
                df.index = pd.to_datetime(df.index)
                df = df.resample("1h").mean().fillna(0.0)
                ema6 = float(df["score"].ewm(span=6, adjust=False).mean().iloc[-1]) if len(df) else 0.0
                ema24 = float(df["score"].ewm(span=24, adjust=False).mean().iloc[-1]) if len(df) else 0.0
                cnt_val = int((df["score"] != 0).astype(int).rolling(24, min_periods=1).sum().iloc[-1])
            else:
                ema6 = ema24 = 0.0
                cnt_val = 0

            topic_bias = keyword_hits = 0.0
            source_rel_avg = novelty_avg = 0.5
            if erows:
                feats = []
                for (fj,) in erows:
                    try:
                        feats.append(json.loads(fj or "{}"))
                    except Exception:
                        feats.append({})
                topic_bias = float(np.mean([f.get("topic_bias", 0.0) for f in feats]))
                keyword_hits = float(np.mean([f.get("keyword_hits", 0.0) for f in feats]))
                source_rel_avg = float(np.mean([f.get("source_reliability", 0.5) for f in feats]))
                novelty_avg = float(np.mean([f.get("novelty", 0.5) for f in feats]))

            market_ret_1h = market_ret_6h = market_ret_24h = market_vol = 0.0
            try:
                kl = klines_close_prices(full, days=2, interval="1h")
                if len(kl) >= 30:
                    closes = np.array([k[1] for k in kl], dtype=float)
                    market_ret_1h = float(closes[-1] / closes[-2] - 1.0)
                    market_ret_6h = float(closes[-1] / closes[-7] - 1.0)
                    market_ret_24h = float(closes[-1] / closes[-25] - 1.0)
                    rr = closes[1:] / closes[:-1] - 1.0
                    market_vol = float(np.std(rr[-24:]))
            except Exception:
                pass

            pred_val: Optional[float] = None
            prob_up = 0.5
            conf_rank = "low"
            conf_score = 0.1
            direction = "up"
            model_score: Optional[float] = None
            reason_codes = [
                f"sent_ema6={ema6:+.3f}",
                f"sent_ema24={ema24:+.3f}",
                f"news_cnt={cnt_val}",
                f"topic_bias={topic_bias:+.3f}",
                f"source_rel={source_rel_avg:.2f}",
                f"novelty={novelty_avg:.2f}",
            ]

            feature_map = {
                "ema6": ema6,
                "ema24": ema24,
                "cnt": cnt_val,
                "ret_1h": market_ret_1h,
                "ret_6h": market_ret_6h,
                "ret_24h": market_ret_24h,
                "vol_24h": market_vol,
                "topic_bias": topic_bias,
                "keyword_hits": keyword_hits,
                "source_reliability": source_rel_avg,
                "novelty": novelty_avg,
            }

            meta = db.execute(
                "SELECT trained_at, n_samples, quality_score, path FROM models_horizon WHERE coin=? AND horizon_hours=24",
                (coin,),
            ).fetchone()

            if meta:
                _, n_samples, quality_score, mdl_path = meta
                model_score = quality_score
                try:
                    with open(mdl_path, "rb") as f:
                        obj = pickle.load(f)
                    if isinstance(obj, dict) and "model" in obj:
                        mdl = obj["model"]
                        feats = obj.get("features", ["ema6", "ema24", "cnt"])
                    else:
                        mdl = obj
                        feats = ["ema6", "ema24", "cnt"]

                    x = np.array([[feature_map.get(name, 0.0) for name in feats]], dtype=float)
                    yhat = float(mdl.predict(x)[0])
                    if not math.isfinite(yhat):
                        raise ValueError("non-finite yhat")
                    pred_val = yhat
                    prob_up = logistic(yhat * 35.0)
                    conf_score = min(0.99, max(0.05, abs(yhat) * 30.0 + 0.2 * source_rel_avg))
                    reason_codes.extend([
                        f"ret_6h={market_ret_6h:+.4f}",
                        f"ret_24h={market_ret_24h:+.4f}",
                        f"vol_24h={market_vol:.4f}",
                    ])
                except Exception as e:
                    print(f"Predict {coin} failed:", e)

            if pred_val is None:
                s = 0.6 * ema6 + 0.25 * ema24 + 0.15 * topic_bias
                pred_val = float(s * 0.02 + 0.2 * market_ret_6h + 0.005 * (source_rel_avg - 0.5))
                prob_up = logistic((ema6 + ema24 + topic_bias + market_ret_6h * 5.0) * 2.0)
                conf_score = min(0.8, max(0.05, abs(s) + 0.15 * source_rel_avg))
                reason_codes.append("heuristic-fallback")

            direction = "up" if pred_val >= 0 else "down"
            if conf_score >= 0.70 and sample >= 6:
                conf_rank = "high"
            elif conf_score >= 0.30:
                conf_rank = "med"
            else:
                conf_rank = "low"

            out.append(
                {
                    "symbol": coin,
                    "direction": direction,
                    "confidence": conf_rank,
                    "confidence_score": round(conf_score, 3),
                    "probability_up": round(prob_up, 3),
                    "expected_move_pct": round(pred_val * 100.0, 3),
                    "score": round(pred_val, 4),
                    "sample_size": sample,
                    "model_version": MODEL_VERSION,
                    "model_quality_score": model_score,
                    "reason_codes": reason_codes[:8],
                }
            )

            save_metric(
                "prediction",
                now.isoformat(),
                coin,
                {
                    "window_hours": window_hours,
                    "direction": direction,
                    "confidence": conf_rank,
                    "confidence_score": conf_score,
                    "probability_up": prob_up,
                    "expected_move": pred_val,
                    "sample_size": sample,
                    "model_quality_score": model_score,
                    "reason_codes": reason_codes[:8],
                },
            )

            try:
                if MEM is not None:
                    MEM.add_or_skip(
                        id=f"pred-{coin}-{now.isoformat()}",
                        text=(
                            f"{coin} {direction} prob={prob_up:.2f} exp={pred_val:+.4f} "
                            f"conf={conf_rank} reasons={','.join(reason_codes[:3])}"
                        ),
                        ts=now.isoformat(),
                        coins=[coin],
                    )
            except Exception as me:
                print("Memory add error (prediction):", me)

    out.sort(
        key=lambda d: (
            {"high": 2, "med": 1, "low": 0}[d["confidence"]],
            d["confidence_score"],
            abs(d["score"]),
        ),
        reverse=True,
    )
    return {
        "asof": now.isoformat(),
        "window_hours": window_hours,
        "model_version": MODEL_VERSION,
        "coins": out,
    }


# =============================================================================
# Single-page UI
# =============================================================================
@app.get("/", response_class=HTMLResponse)
def home():
    w = window_ends()
    alerts_day = db_get_alerts_between(w["day"], w["now"])
    alerts_week = db_get_alerts_between(w["week"], w["now"])
    alerts_month = db_get_alerts_between(w["month"], w["now"])
    alerts_year = db_get_alerts_between(w["year"], w["now"])
    news = db_get_news(30)
    preds = predictions_api(window_hours=48)

    def render_alerts(items: List[dict]) -> str:
        if not items:
            return '<div class="muted">No alerts in this period yet.</div>'
        rows = []
        for a in items[:100]:
            rows.append(
                f"""
            <div class="card">
              <div class="row">
                <div class="pill">{a['coin']}</div>
                <div class="conf {a['confidence'].lower()}">{a['confidence']}</div>
                <div class="score">{float(a['score']):+0.2f}</div>
              </div>
              <div class="title">{a['title']}</div>
              <div class="meta"><a class="link" href="{a['url']}">open</a> • {a['ts']}</div>
              <div class="reasons">{a['reasons']}</div>
            </div>
            """
            )
        return "\n".join(rows)

    alerts_sections = f"""
      <div class="tabs">
        <button class="tab active" data-pane="pane-day">Day</button>
        <button class="tab" data-pane="pane-week">Week</button>
        <button class="tab" data-pane="pane-month">Month</button>
        <button class="tab" data-pane="pane-year">Year</button>
      </div>
      <div id="pane-day" class="pane active">{render_alerts(alerts_day)}</div>
      <div id="pane-week" class="pane">{render_alerts(alerts_week)}</div>
      <div id="pane-month" class="pane">{render_alerts(alerts_month)}</div>
      <div id="pane-year" class="pane">{render_alerts(alerts_year)}</div>
    """

    news_rows = []
    for n in news:
        news_rows.append(
            f"""
        <div class="card alt">
          <div class="title">{n['title']}</div>
          <div class="meta"><a class="link" href="{n['url']}">open</a> • {n['ts']}</div>
        </div>
        """
        )
    news_html = "\n".join(news_rows) if news_rows else '<div class="muted">Fetching RSS…</div>'

    def render_predictions(preds_dict: dict) -> str:
        coins = preds_dict.get("coins", [])
        if not coins:
            return '<div class="muted">No predictions yet — gathering data…</div>'
        cards = []
        for c in coins:
            base = c["symbol"]
            meta = COIN_META.get(base, {"icon": base[:1], "color": "#444"})
            arrow = "⬆️" if c["direction"] == "up" else "⬇️"
            conf_cls = {"low": "low", "med": "med", "high": "high"}[c["confidence"]]
            score = f"{c['score']:+.3f}"
            sample = c.get("sample_size", 0)
            cards.append(
                f"""
              <div class="card pred">
                <div class="row" style="justify-content:space-between">
                  <div class="row" style="gap:10px">
                    <div class="avatar" style="background:{meta['color']}">{meta['icon']}</div>
                    <div class="title">{base} {arrow}</div>
                  </div>
                  <div class="conf {conf_cls}">{c['confidence'].capitalize()}</div>
                </div>
                <div class="meta">Score: <b>{score}</b> • Sources: {sample}</div>
              </div>
            """
            )
        return "\n".join(cards)

    preds_html = render_predictions(preds)

    html_template = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = (
        html_template.replace("__ALERTS_SECTIONS__", alerts_sections)
        .replace("__NEWS_HTML__", news_html)
        .replace("__PREDS_HTML__", preds_html)
        .replace("__FEEDS_COUNT__", str(len(FEEDS)))
        .replace("__ORDER__", json.dumps(TICKER_SYMBOLS))
        .replace("__COIN_META__", json.dumps(COIN_META))
        .replace("__MODEL_VERSION__", MODEL_VERSION)
        .replace("__APP_VERSION__", VERSION)
    )
    return HTMLResponse(html)


# =============================================================================
# (A long tail of comments / inline docs to help you navigate the file)
# =============================================================================
#
# 1) System overview
#    - rss_loop(): collects headlines, stores in DB, writes per-coin sentiment,
#      and pushes "documents" into the semantic memory (MEM.add_or_skip).
#    - price_loop(): fills PRICES dict for the moving ticker at the bottom.
#    - alert_generation_loop(): quick heuristic for demo alerts.
#    - daily_trainer_loop(): trains a Ridge model per-coin every ~6h and stores
#      model metadata + .pkl into /data/models.
#    - predictions_api: aggregates features for the last N hours and does:
#        a) ML prediction if model exists and loads
#        b) else heuristic fallback
#      It also logs a metric row and drops a small text into the memory.
#
# 2) Memory integration
#    - MemoryIndex is optional; if import fails or model download is blocked,
#      app prints a clear message and continues without memory features.
#    - Use /memory/search?q=... to test that memory is indexing headlines and
#      prediction lines. It returns early list of documents with metadata.
#
# 3) FinBERT integration
#    - If FinBERT is unavailable, we fall back to a simple lexicon-based
#      sentiment classifier. This keeps the pipeline alive in offline setups.
#
# 4) Docker bind mount reminder
#    - Always run the container with `-v "C:\crypto-intel-mini\data:/data"`
#      on Windows so that your database and models persist and you can see them.
#
# 5) Troubleshooting quick checks
#    - If /memory/search returns {"hits":[]}, give it a couple minutes
#      (RSS loop runs every 5 minutes) and ensure memory started in logs:
#        [Memory] semantic index ready.
#    - If /predictions returns little data at first, wait for some headlines
#      to build up sentiments. The trainer runs every 6 hours by default.
#    - If you see "UserWarning: X does not have valid feature names", it's
#      benign — Sklearn compares training df columns vs. inference array.
#
# 6) You can safely extend
#    - Add more feeds in FEEDS
#    - Tweak heuristic thresholds in predictions_api
#    - Add new endpoints that query the DB / memory, etc.
#
# =============================================================================
# End of File
# =============================================================================
