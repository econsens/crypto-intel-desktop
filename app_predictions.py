# app_predictions.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Literal, Dict
from datetime import datetime
import json
import os

router = APIRouter()

PREDICTIONS_PATH = "/data/predictions.json"

class Driver(BaseModel):
    title: str
    url: str
    score: float
    features: Dict[str, float]

class CoinPred(BaseModel):
    symbol: str
    direction: Literal["up","down","neutral"]
    confidence: Literal["low","med","high"]
    net_signal: float
    volume: int
    window_hours: int
    top_drivers: List[Driver]

class Predictions(BaseModel):
    asof: datetime
    coins: List[CoinPred]

@router.get("/predictions", response_model=Predictions)
def get_predictions():
    if not os.path.exists(PREDICTIONS_PATH):
        # serve empty but valid shape
        return Predictions(asof=datetime.utcnow(), coins=[])
    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

@router.get("/predictions/{symbol}", response_model=CoinPred)
def get_prediction(symbol: str):
    if not os.path.exists(PREDICTIONS_PATH):
        raise HTTPException(404, "No predictions yet")
    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    for c in data.get("coins", []):
        if c["symbol"].lower() == symbol.lower():
            return c
    raise HTTPException(404, f"No prediction for {symbol.upper()}")
