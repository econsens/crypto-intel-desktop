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
import random
import sqlite3
import httpx
from hashlib import sha256
from datetime import datetime, timezone, timedelta
from dateutil import parser as dtparser
import pickle
import math
VERSION = "1.1.0"

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

# shared price cache for ticker bar
PRICES: Dict[str, Dict[str, float]] = {}


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
                }
            )
    return items


def rss_loop():
    """Continuously fetch RSS, save to DB, compute per-coin sentiment, and feed memory."""
    while True:
        try:
            for n in fetch_rss_once():
                # 1) Save the news item
                db_add_news(n)

                # 2) Per-coin sentiment for this title
                cs = coins_in_title(n["title"])
                if cs:
                    s = signed_sentiment(n["title"])
                    with sqlite3.connect(DB_PATH) as db:
                        for coin in cs:
                            db.execute(
                                "INSERT OR REPLACE INTO sentiments(nid, coin, ts, score, source) VALUES(?,?,?,?,?)",
                                (
                                    n["id"],
                                    coin,
                                    n["ts"],
                                    float(s),
                                    "finbert" if _FINBERT else "lexicon",
                                ),
                            )

                # 3) Feed the memory index (skip near-duplicates automatically)
                try:
                    if MEM is not None:
                        # Consistent with our MemoryIndex API: add_or_skip(id, text, ts, coins)
                        MEM.add_or_skip(n["id"], n["title"], n["ts"], cs)
                except Exception as me:
                    print("Memory add error (headline):", me)

        except Exception as e:
            print("RSS loop error:", e)

        time.sleep(300)  # every 5 minutes


def alert_generation_loop() -> None:
    """Simple alert scoring (placeholder)."""
    while True:
        try:
            for n in db_get_news(25):
                score = round(random.uniform(-0.3, 1.0), 2)
                coin = random.choice(["BTC", "ETH", "SOL", "BNB", "XRP", "LINK", "ADA"])
                confidence = "High" if score >= 0.75 else "Med" if score >= 0.55 else "Low"
                reasons = []
                if score >= 0.5:
                    reasons.append("Positive keywords")
                if coin in n["title"].upper():
                    reasons.append("Named coin")
                if score < 0:
                    reasons.append("Negative words")
                item = {
                    "id": n["id"],
                    "title": n["title"],
                    "url": n["url"],
                    "coin": coin,
                    "score": score,
                    "confidence": confidence,
                    "ts": n["ts"],
                    "reasons": "; ".join(reasons) if reasons else "Heuristic",
                }
                db_add_alert(item)
        except Exception as e:
            print("Alert loop error:", e)
        time.sleep(60)


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
    Every ~6h: train a Ridge model per coin to predict next-24h return
    from rolling sentiment features. Models saved to /data/models.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    import numpy as np  # noqa: F401
    import pandas as pd

    horizon_hours = 24
    while True:
        try:
            print("Trainer: starting pass…")
            for full in TICKER_SYMBOLS:
                coin = full.replace("USDT", "")
                try:
                    # 1) prices
                    kl = klines_close_prices(full, days=30, interval="1h")
                    if len(kl) < 48:
                        continue
                    dfp = pd.DataFrame(kl, columns=["ts", "close"]).set_index("ts")
                    dfp.index = pd.to_datetime(dfp.index)
                    dfp["ret_next_24h"] = (
                        dfp["close"].pct_change(periods=horizon_hours).shift(-horizon_hours)
                    )

                    # 2) sentiments -> hourly features
                    since_iso = dfp.index.min().isoformat()
                    with sqlite3.connect(DB_PATH) as db:
                        srows = db.execute(
                            "SELECT ts, score FROM sentiments WHERE coin=? AND ts>=? ORDER BY ts ASC",
                            (coin, since_iso),
                        ).fetchall()
                    if not srows:
                        continue

                    dfs = pd.DataFrame(srows, columns=["ts", "score"]).set_index("ts")
                    dfs.index = pd.to_datetime(dfs.index)
                    dfs = dfs.resample("1h").mean().fillna(0.0)
                    dfs["ema6"] = dfs["score"].ewm(span=6, adjust=False).mean()
                    dfs["ema24"] = dfs["score"].ewm(span=24, adjust=False).mean()
                    # rolling count with min_periods=1 so it's never NaN
                    dfs["cnt"] = (dfs["score"] != 0).astype(int).rolling(24, min_periods=1).sum()

                    d = dfp.join(dfs[["ema6", "ema24", "cnt"]], how="left").fillna(0.0)

                    y = d["ret_next_24h"].dropna()
                    X = d.loc[y.index, ["ema6", "ema24", "cnt"]]
                    if len(y) < 48:
                        continue

                    n = len(y)
                    split = int(n * 0.8)
                    Xtr, Xte = X.iloc[:split], X.iloc[split:]
                    ytr, yte = y.iloc[:split], y.iloc[split:]

                    mdl = Ridge(alpha=0.5).fit(Xtr, ytr)
                    yhat = mdl.predict(Xte)
                    r2 = float(r2_score(yte, yhat))

                    art = {
                        "coin": coin,
                        "trained_at": datetime.now(timezone.utc).isoformat(),
                        "horizon_hours": horizon_hours,
                        "n_samples": int(n),
                        "r2": r2,
                        "path": os.path.join(MODEL_DIR, f"{coin}.pkl"),
                    }
                    with open(art["path"], "wb") as f:
                        pickle.dump(mdl, f)

                    with sqlite3.connect(DB_PATH) as db:
                        db.execute(
                            "INSERT OR REPLACE INTO models VALUES(?,?,?,?,?,?)",
                            (
                                art["coin"],
                                art["trained_at"],
                                art["horizon_hours"],
                                art["n_samples"],
                                art["r2"],
                                art["path"],
                            ),
                        )
                    print(f"Trainer: {coin} r2={r2:.3f} n={n}")
                except Exception as e:
                    print(f"Trainer: {coin} failed:", e)
        except Exception as e:
            print("Trainer loop error:", e)
        time.sleep(6 * 3600)


# =============================================================================
# FastAPI app + startup
# =============================================================================
app = FastAPI(title="Crypto Intel")


@app.on_event("startup")
def on_start():
    global MEM

    db_init()

    # Start background loops
    threading.Thread(target=rss_loop, daemon=True).start()
    threading.Thread(target=alert_generation_loop, daemon=True).start()
    threading.Thread(target=price_loop, daemon=True).start()
    try:
        threading.Thread(target=daily_trainer_loop, daemon=True).start()
    except NameError:
        pass

    # Start semantic memory (optional)
    try:
        if MemoryIndex is not None:
            # directory, not DB file
            MEM = MemoryIndex(base_dir=MEM_DIR)
            # use MiniLM sentence transformer
            MEM.start(model_name="sentence-transformers/all-MiniLM-L6-v2")
            print("[Memory] semantic index ready.")
        else:
            print("[Memory] package not available; skipping.")
    except Exception as e:
        MEM = None
        print("[Memory] failed to start:", e)


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


@app.get("/news")
def news_api():
    return JSONResponse(db_get_news(100))


@app.get("/predictions")
def predictions_api(window_hours: int = 48):
    """
    Return per-coin prediction based on trained Ridge model if available.
    Fallback: recent sentiment aggregates -> heuristic.
    Also logs a prediction metric & drops a memory line.
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
            sample = len(rows)

            # Aggregate features
            if rows:
                df = pd.DataFrame(rows, columns=["ts", "score"]).set_index("ts")
                df.index = pd.to_datetime(df.index)
                # resample to hourly mean to smooth spikes
                df = df.resample("1h").mean().fillna(0.0)

                # safe EMA values (even with few samples)
                ema6_series = df["score"].ewm(span=6, adjust=False).mean()
                ema24_series = df["score"].ewm(span=24, adjust=False).mean()
                ema6 = float(ema6_series.iloc[-1]) if len(ema6_series) else 0.0
                ema24 = float(ema24_series.iloc[-1]) if len(ema24_series) else 0.0

                # rolling count (never NaN thanks to min_periods=1)
                cnt_val = int(
                    (df["score"] != 0).astype(int).rolling(24, min_periods=1).sum().iloc[-1]
                )
            else:
                ema6 = ema24 = 0.0
                cnt_val = 0

            # ML or heuristic
            pred_val: Optional[float] = None
            conf_rank = "low"
            direction = "up"
            model_r2: Optional[float] = None

            meta = db.execute(
                "SELECT trained_at, horizon_hours, n_samples, r2, path FROM models WHERE coin=?",
                (coin,),
            ).fetchone()

            if meta:
                _, _, n_samples, r2, mdl_path = meta
                model_r2 = r2
                try:
                    with open(mdl_path, "rb") as f:
                        mdl = pickle.load(f)
                    x = np.array([[ema6, ema24, cnt_val]])
                    yhat = float(mdl.predict(x)[0])  # next 24h return prediction
                    if not math.isfinite(yhat):
                        raise ValueError("non-finite yhat")
                    pred_val = yhat
                    mag = abs(yhat)
                    if mag > 0.025 and (sample >= 6 or n_samples >= 100):
                        conf_rank = "high"
                    elif mag > 0.010:
                        conf_rank = "med"
                    else:
                        conf_rank = "low"
                    direction = "up" if yhat >= 0 else "down"
                except Exception as e:
                    print(f"Predict {coin} failed:", e)

            if pred_val is None:
                # Heuristic fallback
                s = 0.6 * ema6 + 0.4 * ema24
                direction = "up" if s >= 0 else "down"
                mag = abs(s)
                if mag > 0.40 and sample >= 6:
                    conf_rank = "high"
                elif mag > 0.15:
                    conf_rank = "med"
                else:
                    conf_rank = "low"
                pred_val = float(s * 0.02)

            out.append(
                {
                    "symbol": coin,
                    "direction": direction,  # "up" | "down"
                    "confidence": conf_rank,  # "high" | "med" | "low"
                    "score": round(pred_val, 3),
                    "sample_size": sample,
                }
            )

            # Save a metric row
            save_metric(
                "prediction",
                now.isoformat(),
                coin,
                {
                    "window_hours": window_hours,
                    "direction": direction,
                    "confidence": conf_rank,
                    "score": pred_val,
                    "sample_size": sample,
                    "model_r2": model_r2,
                },
            )

            # Drop a memory observation (optional)
            try:
                if MEM is not None:
                    # Use the "headline-like" API we defined: add_or_skip(id, text, ts, coins)
                    MEM.add_or_skip(
                        id=f"pred-{coin}-{now.isoformat()}",
                        text=f"Predicted {direction} ({conf_rank}) with score {pred_val:+.3f} over {window_hours}h",
                        ts=now.isoformat(),
                        coins=[coin],
                    )
            except Exception as me:
                print("Memory add error (prediction):", me)

    # Sort: confidence first, then magnitude
    out.sort(
        key=lambda d: ({"high": 2, "med": 1, "low": 0}[d["confidence"]], abs(d["score"])),
        reverse=True,
    )
    return {"asof": now.isoformat(), "window_hours": window_hours, "coins": out}



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
    preds = load_predictions()

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

    html = r"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Crypto Intel</title>
  <style>
    :root {
      --bg: #0b1220; --panel: #0f172a; --panel-2: #0c1526;
      --card: #1f2937; --card-alt: #111827; --text: #e5e7eb;
      --muted: #94a3b8; --link: #60a5fa; --good: #22c55e; --bad: #ef4444;
      --shadow: rgba(0,0,0,0.25); --track: #0a0f1a; --thumb: #2a3342; --thumbh: #3a4558;
      --side: 280px; --footer-h: 48px;
    }
    * { box-sizing:border-box }
    body {
      margin:0; background:var(--bg); color:var(--text); font-family:Segoe UI, Arial, sans-serif;
      overflow-y:scroll; padding-bottom: calc(var(--footer-h) + 12px);
    }
    * { scrollbar-width:thin; scrollbar-color:var(--thumb) var(--track) }
    ::-webkit-scrollbar{ width:10px; height:10px }
    ::-webkit-scrollbar-track{ background:var(--track) }
    ::-webkit-scrollbar-thumb{ background:var(--thumb); border-radius:8px; border:2px solid var(--track) }
    ::-webkit-scrollbar-thumb:hover{ background:var(--thumbh) }

    .layout { display:grid; grid-template-columns: var(--side) 1fr 300px; min-height:100vh }
    .sidebar { background:var(--panel); height:100vh; overflow-y:auto; padding:10px; position:sticky; top:0 }
    .content { padding:20px 24px }
    .right { background:var(--panel); padding:16px }

    .h2 { font-size:22px; margin:6px 0 10px 0 }
    .card { background:var(--card); border-radius:14px; padding:14px; margin:10px 0; box-shadow:0 2px 6px var(--shadow) }
    .card.alt { background:var(--card-alt) }
    .row { display:flex; gap:8px; align-items:center; margin-bottom:6px }
    .pill { background:rgba(255,255,255,.08); padding:3px 8px; border-radius:999px; font-size:12px }
    .conf.low{ color:#f59e0b } .conf.med{ color:#22d3ee } .conf.high{ color:#22c55e }
    .score{ font-weight:700 }
    .title{ font-weight:600 }
    .meta{ font-size:12px; opacity:.8; margin-top:4px }
    .reasons{ font-size:12px; opacity:.82; margin-top:4px }
    .link{ color:var(--link) }
    .tabs{ display:flex; gap:8px; margin:4px 0 8px 0 }
    .tab{ background:var(--card-alt); color:var(--text); border:none; padding:6px 10px; border-radius:10px; cursor:pointer }
    .tab.active{ background:var(--card); font-weight:600 }
    .pane{ display:none } .pane.active{ display:block }

    .avatar{ width:24px; height:24px; border-radius:999px; display:inline-flex; align-items:center; justify-content:center; font-size:14px; color:#fff; font-weight:700 }

    .footer{ position:fixed; left:0; right:0; bottom:0; height:var(--footer-h); background:var(--panel-2); border-top:1px solid rgba(255,255,255,.06); z-index:10 }
    .ticker{ overflow:hidden; white-space:nowrap; width:100%; height:100% }
    .track{ display:inline-flex; gap:18px; padding:10px; animation:scroll 35s linear infinite }
    @keyframes scroll { 0%{transform:translateX(0)} 100%{transform:translateX(-50%)} }
    .tick{ display:inline-flex; gap:10px; padding:6px 12px; background:var(--card); border-radius:999px; align-items:center }
    .tick .avatar{ width:18px; height:18px; font-size:12px }
    .tick .sym{ font-weight:800; letter-spacing:.5px }
    .tick .val{ font-weight:700 }
    .tick.pos .val{ color:var(--good) }
    .tick.neg .val{ color:var(--bad) }
  </style>
  <script>
    const ORDER = __ORDER__;
    const COIN_META = __COIN_META__;

    function fmt(v){ return (v<10)? v.toFixed(4) : v.toFixed(2); }

    function renderTicker(data){
      const track = document.getElementById('ticker-track');
      if(!track) return;
      let html = '';
      for(const full of ORDER){
        const base = full.replace('USDT','');
        const p = data[full];
        const meta = COIN_META[base] || {icon: base[0], color:'#555'};
        if(!p){
          html += `<div class="tick"><span class="avatar" style="background:${meta.color}">${meta.icon}</span><span class="sym">${base}</span><span class="val">…</span></div>`;
          continue;
        }
        const pct  = (p.changePct>=0?'+':'') + p.changePct.toFixed(2) + '%';
        const last = '$' + fmt(p.last);
        const cls  = p.changePct>=0 ? 'pos' : 'neg';
        html += `
          <div class="tick ${cls}">
            <span class="avatar" style="background:${meta.color}">${meta.icon}</span>
            <span class="sym">${base}</span>
            <span class="val">${pct} / ${last}</span>
          </div>`;
      }
      track.innerHTML = html + html;
    }

    function wireTabs(){
      const tabs = document.querySelectorAll('.tab');
      tabs.forEach(t => t.addEventListener('click', () => {
        tabs.forEach(x=>x.classList.remove('active'));
        t.classList.add('active');
        document.querySelectorAll('.pane').forEach(p=>p.classList.remove('active'));
        document.getElementById(t.dataset.pane).classList.add('active');
      }));
    }

    async function fetchPrices(){
      try{
        const r = await fetch('/prices', {cache:'no-store'});
        renderTicker(await r.json());
      }catch(e){}
    }

    window.addEventListener('DOMContentLoaded', ()=>{
      wireTabs();
      fetchPrices();
      setInterval(fetchPrices, 12000);
    });
  </script>
</head>
<body>
  <div class="layout">
    <div class="sidebar">
      <div style="font-weight:700; padding:8px 6px;">Menu</div>
      <a class="link" href="#alerts" style="display:block; padding:6px 10px;">🔔 Alerts</a>
      <a class="link" href="#headlines" style="display:block; padding:6px 10px;">📰 Headlines</a>
      <a class="link" href="#predictions" style="display:block; padding:6px 10px;">📈 Predictions</a>
    </div>

    <div class="content">
      <h2 id="alerts" class="h2">🔔 Alerts</h2>
      __ALERTS_SECTIONS__

      <h2 id="headlines" class="h2" style="margin-top:18px">📰 Latest headlines</h2>
      __NEWS_HTML__

      <h2 id="predictions" class="h2" style="margin-top:18px">📈 Predictions</h2>
      __PREDS_HTML__
    </div>

    <div class="right">
      <h3>Status</h3>
      <div style="opacity:.8">Local only (127.0.0.1)</div>
      <div style="opacity:.8">Updates every 5 minutes</div>
      <div style="opacity:.8">Feeds: __FEEDS_COUNT__</div>
      <div style="opacity:.8">Version: __APP_VERSION__</div>
      <div class="footer">
        <div class="ticker"><div id="ticker-track" class="track"></div></div>
      </div>
    </div>
  </div>
</body>
</html>
"""
    html = (
        html.replace("__ALERTS_SECTIONS__", alerts_sections)
        .replace("__NEWS_HTML__", news_html)
        .replace("__PREDS_HTML__", preds_html)
        .replace("__FEEDS_COUNT__", str(len(FEEDS)))
        .replace("__ORDER__", json.dumps(TICKER_SYMBOLS))
        .replace("__COIN_META__", json.dumps(COIN_META))
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
