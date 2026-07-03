"""Post-trade attribution (spec §17).

After a prediction window closes, decompose the realised move into market /
sector / company-specific components (via beta) and judge whether the call was
right and well-reasoned. Produces a short 'lesson' for the model-improvement loop.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.timestamp_utils import to_utc


def _ret_over(prices: pd.DataFrame | None, start, end) -> float:
    if prices is None or prices.empty:
        return float("nan")
    p = prices.sort_index()["close"].astype(float)
    try:
        c0 = float(p.loc[p.index <= start].iloc[-1])
        c1 = float(p.loc[p.index <= end].iloc[-1])
        return (c1 - c0) / c0
    except (IndexError, KeyError):
        return float("nan")


def run_attribution(
    signal: dict[str, Any],
    prices: pd.DataFrame,
    context: dict[str, pd.DataFrame],
    ticker_cfg: dict[str, Any],
    horizon_bars: int = 1,
) -> dict[str, Any]:
    """Attribute the realised move to market / sector / company-specific drivers."""
    ts = to_utc(pd.Timestamp(signal["timestamp"]))
    idx = prices.sort_index().index
    after = idx[idx > ts]
    if len(after) < horizon_bars:
        return {"status": "pending"}
    end = after[horizon_bars - 1]

    actual = _ret_over(prices, ts, end)
    market_sym = next((s.replace("^", "_") for s in ticker_cfg.get("market_etfs", [])
                       if s.replace("^", "_") in context), None)
    sector_sym = next((s.replace("^", "_") for s in ticker_cfg.get("sector_etfs", [])
                       if s.replace("^", "_") in context), None)
    market_ret = _ret_over(context.get(market_sym), ts, end) if market_sym else float("nan")
    sector_ret = _ret_over(context.get(sector_sym), ts, end) if sector_sym else float("nan")

    beta = float(signal.get("market_beta", 1.0) or 1.0)
    market_component = beta * market_ret if market_ret == market_ret else 0.0
    sector_component = (sector_ret - market_ret) if (sector_ret == sector_ret and market_ret == market_ret) else 0.0
    company_component = actual - market_component - sector_component if actual == actual else float("nan")

    # Dominant driver.
    comps = {
        "market_driven": abs(market_component),
        "sector_driven": abs(sector_component),
        "company_specific": abs(company_component) if company_component == company_component else 0.0,
    }
    dominant = max(comps, key=comps.get)

    decision = signal.get("decision")
    correct = (actual < 0) if decision in ("SHORT", "WATCH") else (actual >= -0.005)
    lesson = _lesson(decision, correct, dominant, signal)

    return {
        "status": "complete",
        "actual_return": float(actual) if actual == actual else None,
        "market_component": float(market_component),
        "sector_component": float(sector_component),
        "company_component": float(company_component) if company_component == company_component else None,
        "dominant_driver": dominant,
        "prediction_correct": bool(correct),
        "lesson": lesson,
    }


def _lesson(decision, correct, dominant, signal) -> str:
    pv = float(signal.get("price_volume_confirmation", 0.0) or 0.0)
    if decision == "SHORT" and correct:
        return f"Correct short; move was {dominant.replace('_', ' ')}. Confirmation held (pv={pv:.2f})."
    if decision == "SHORT" and not correct:
        return (f"Short failed; move was {dominant.replace('_', ' ')}. "
                "Consider raising the confirmation threshold or reducing size.")
    if decision == "WATCH":
        return f"Watch resolved {'down' if correct else 'up'}; primarily {dominant.replace('_', ' ')}."
    if decision == "NO TRADE" and correct:
        return "No-trade correctly avoided a non-event — capital preserved."
    return f"No-trade missed a {dominant.replace('_', ' ')} move; review gate thresholds."
