# tasks_predictions.py
import math, json, os
from datetime import datetime, timedelta
from typing import List, Dict, Any

PREDICTIONS_PATH = "/data/predictions.json"
WINDOW = timedelta(hours=24)

# Coins you already use (taken from your app’s bottom bar & UI)
COIN_SYNONYMS = {
    # BTC
    "bitcoin": "BTC", "btc": "BTC",
    # ETH
    "ethereum": "ETH", "eth": "ETH",
    # SOL
    "solana": "SOL", "sol": "SOL",
    # XRP
    "xrp": "XRP", "ripple": "XRP",
    # ADA
    "ada": "ADA", "cardano": "ADA",
    # DOGE
    "doge": "DOGE", "dogecoin": "DOGE",
    # MATIC
    "matic": "MATIC", "polygon": "MATIC",
    # DOT
    "dot": "DOT", "polkadot": "DOT",
    # LINK
    "link": "LINK", "chainlink": "LINK",
    # BNB
    "bnb": "BNB",
    # AVAX
    "avax": "AVAX", "avalanche": "AVAX",
    # TON
    "ton": "TON", "toncoin": "TON"
}

SOURCE_WEIGHTS = {
    "reuters": 1.2,
    "bloomberg": 1.2,
    "coindesk": 1.1,
    "cointelegraph": 1.05
}
DEFAULT_SOURCE_WEIGHT = 1.0

EVENT_POS = {"partnership","integration","listing","launch","approval","upgrade","milestone","expands","onboards"}
EVENT_NEG = {"hack","exploit","lawsuit","outage","ban","halt","pauses","fined"}

def recency_decay(dt, now):
    hours = (now - dt).total_seconds()/3600.0
    tau = 12.0
    return math.exp(-hours/tau)

# quick, deterministic sentiment (replace with transformer later if you want)
def quick_sentiment(title: str) -> float:
    t = title.lower()
    pos = any(k in t for k in ["surge","rally","partnership","approval","expands","record","growth","upgrade","launch"])
    neg = any(k in t for k in ["drop","plunge","lawsuit","hack","outage","ban","halts","crash","exploit"])
    if pos and not neg: return 0.6
    if neg and not pos: return -0.6
    return 0.0

def event_boost(title: str, sentiment: float) -> float:
    t = title.lower()
    if any(w in t for w in EVENT_POS):
        return 1.2 if sentiment > 0 else 0.9
    if any(w in t for w in EVENT_NEG):
        return 0.8 if sentiment > 0 else 1.2
    return 1.0

def extract_coins(title: str) -> List[str]:
    t = f" {title.lower()} "
    out = set()
    for k, v in COIN_SYNONYMS.items():
        if f" {k} " in t:
            out.add(v)
    return list(out)

def _iter_headlines_from_data_dir() -> List[Dict[str, Any]]:
    """
    Read whatever your backend already saves for the UI.
    This function is tolerant:
      - /data/alerts.json  (list of {time,title,url,source,...})
      - /data/headlines.json
      - If neither exists, returns [].
    """
    paths = ["/data/alerts.json", "/data/headlines.json"]
    items: List[Dict[str, Any]] = []
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        items.extend(data)
                    elif isinstance(data, dict) and "items" in data:
                        items.extend(data["items"])
            except Exception:
                pass
    # Expect fields: time (ISO), title, url, source
    cleaned = []
    for it in items:
        t = it.get("time") or it.get("published_at") or it.get("date")
        title = it.get("title") or it.get("headline") or ""
        url = it.get("url") or it.get("link") or ""
        source = (it.get("source") or it.get("publisher") or "").lower()
        if not t or not title:
            continue
        try:
            _ = datetime.fromisoformat(t.replace("Z",""))
        except Exception:
            continue
        cleaned.append({"time": t.replace("Z",""), "title": title, "url": url, "source": source})
    return cleaned

def compute_predictions(headlines: List[Dict[str, Any]]) -> Dict[str, Any]:
    now = datetime.utcnow()
    per_coin: Dict[str, Dict[str, Any]] = {}
    for h in headlines:
        try:
            dt = datetime.fromisoformat(h["time"])
        except Exception:
            continue
        if now - dt > WINDOW:
            continue
        title = h["title"]
        coins = extract_coins(title)
        if not coins:
            continue
        source = h.get("source","").lower()
        sent = quick_sentiment(title)
        sw   = SOURCE_WEIGHTS.get(source, DEFAULT_SOURCE_WEIGHT)
        rd   = recency_decay(dt, now)
        eb   = event_boost(title, sent)
        score = sent * sw * rd * eb
        for c in coins:
            pc = per_coin.setdefault(c, {"sum":0.0, "vol":0, "drivers":[]})
            pc["sum"] += score
            pc["vol"] += 1
            pc["drivers"].append({
                "title": title,
                "url": h.get("url",""),
                "score": round(score, 4),
                "features": {
                    "sentiment": sent,
                    "source_weight": sw,
                    "recency_decay": round(rd, 4),
                    "event_boost": eb
                }
            })

    coins_out = []
    T, T_MED, T_HIGH = 0.8, 1.2, 2.0
    V_MED, V_HIGH = 3, 6

    for c, v in per_coin.items():
        s = v["sum"]; vol = v["vol"]
        if s >  T:   direction="up"
        elif s < -T: direction="down"
        else:       direction="neutral"

        abs_s = abs(s)
        if abs_s >= T_HIGH and vol >= V_HIGH: conf="high"
        elif abs_s >= T_MED and vol >= V_MED: conf="med"
        else: conf="low"

        drivers = sorted(v["drivers"], key=lambda d: abs(d["score"]), reverse=True)[:3]
        coins_out.append({
            "symbol": c,
            "direction": direction,
            "confidence": conf,
            "net_signal": round(s, 3),
            "volume": vol,
            "window_hours": int(WINDOW.total_seconds()/3600),
            "top_drivers": drivers
        })

    return {
        "asof": now.isoformat()+"Z",
        "coins": sorted(coins_out, key=lambda x: -abs(x["net_signal"]))
    }

def save_predictions(data: Dict[str, Any]):
    os.makedirs(os.path.dirname(PREDICTIONS_PATH), exist_ok=True)
    with open(PREDICTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def prediction_job():
    headlines = _iter_headlines_from_data_dir()
    data = compute_predictions(headlines)
    save_predictions(data)
