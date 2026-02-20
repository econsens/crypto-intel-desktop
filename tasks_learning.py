# tasks_learning.py — Phase 3.1 “Learning” task
# Compares current /predictions direction against realized price move,
# then appends a summary row to data/learning_log.json

import os, json, time, requests
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

API = "http://127.0.0.1:8000"
BINANCE = "https://api.binance.com"

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
LOG_PATH = os.path.join(DATA_DIR, "learning_log.json")


def get_predictions(window_hours: int = 24) -> List[Dict[str, Any]]:
    """Fetch predictions and normalize to: symbol, pred_sign (+1/-1), score."""
    r = requests.get(f"{API}/predictions", params={"window_hours": window_hours}, timeout=10)
    r.raise_for_status()
    data = r.json()
    coins = data.get("coins", [])

    out = []
    for p in coins:
        # If server ever returned a string, try to parse it
        if isinstance(p, str):
            try:
                p = json.loads(p)
            except Exception:
                continue

        sym = p.get("symbol") or p.get("coin") or ""
        direction = p.get("direction")
        score = float(p.get("score", 0) or 0)

        if not sym or direction not in ("up", "down"):
            continue

        pred_sign = 1 if direction == "up" else -1
        out.append({"symbol": sym, "pred_sign": pred_sign, "score": score})

    return out


def binance_return(base_symbol: str, hours: int = 24) -> Optional[float]:
    """Compute % return over the last `hours` using 1h klines."""
    symbol = f"{base_symbol}USDT"
    end = int(time.time() * 1000)
    start = end - hours * 60 * 60 * 1000

    url = f"{BINANCE}/api/v3/klines"
    params = {"symbol": symbol, "interval": "1h", "startTime": start, "endTime": end}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    kl = r.json()
    if len(kl) < 2:
        return None

    first_close = float(kl[0][4])
    last_close = float(kl[-1][4])
    return (last_close / first_close) - 1.0  # e.g. +0.012 = +1.2%


def compare_predictions(window_hours: int = 24) -> Optional[Dict[str, Any]]:
    preds = get_predictions(window_hours)
    rows = []

    for p in preds:
        realized = binance_return(p["symbol"], hours=window_hours)
        if realized is None:
            continue
        actual_sign = 1 if realized >= 0 else -1
        correct = 1 if actual_sign == p["pred_sign"] else 0
        rows.append({
            "coin": p["symbol"],
            "pred_sign": p["pred_sign"],
            "realized_return": realized,
            "correct": correct
        })

    if not rows:
        return None

    acc = sum(r["correct"] for r in rows) / len(rows)
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "window_hours": window_hours,
        "n": len(rows),
        "acc": acc,
        "by_coin": rows
    }


def append_log(entry: Dict[str, Any]) -> None:
    log = []
    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                log = json.load(f)
        except Exception:
            log = []
    log.append(entry)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    try:
        result = compare_predictions(24)  # you can change to 12/48 if you want
        if result:
            append_log(result)
            print(f"Saved learning entry: acc={result['acc']:.3f} over {result['n']} coins")
        else:
            print("No data to log yet (try again later).")
    except Exception as e:
        print("Learning task failed:", e)
