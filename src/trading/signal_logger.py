"""Prediction ledger (spec §15) — every signal logged BEFORE its outcome.

This is the accountability backbone (spec §34): each prediction is timestamped,
stored with its full model/decision context, and later reconciled against the
realised return. Backed by a CSV (data/predictions/prediction_ledger.csv) for
MVP simplicity; can move to SQLite later.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config_loader import data_dir
from src.utils.logging_utils import get_logger
from src.utils.timestamp_utils import now_utc, to_utc

log = get_logger("trading.ledger")

LEDGER_COLUMNS = [
    "signal_id", "timestamp", "ticker", "current_price", "prediction_horizon",
    "xgb_downside_probability", "adjusted_downside_probability", "xgb_confidence",
    "garch_volatility_forecast", "garch_volatility_regime", "var_estimate",
    "expected_shortfall_estimate", "quantile_5pct_return", "quantile_10pct_return",
    "model_agreement_score", "data_confidence_score", "base_downside_risk_score",
    "adjusted_downside_risk_score", "uncertainty_penalty", "price_volume_confirmation",
    "news_shift_logodds", "expected_edge_bps",
    "decision", "recommended_position_size", "position_notional", "stop_loss", "take_profit",
    "top_3_signal_drivers", "news_catalysts", "market_regime",
    "actual_return_12h", "actual_return_24h", "prediction_correct",
    "paper_trade_result", "post_trade_attribution", "created_at", "updated_at",
]


def ledger_path() -> Path:
    p = data_dir() / "predictions"
    p.mkdir(parents=True, exist_ok=True)
    return p / "prediction_ledger.csv"


def load_ledger() -> pd.DataFrame:
    path = ledger_path()
    if not path.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    return pd.read_csv(path)


def _save_ledger(df: pd.DataFrame) -> None:
    df.to_csv(ledger_path(), index=False)


def log_prediction(decision: dict[str, Any], sizing: dict[str, Any] | None = None) -> str:
    """Append a prediction row (outcome fields left blank). Returns signal_id."""
    sizing = sizing or {}
    signal_id = uuid.uuid4().hex[:12]
    now = now_utc().isoformat()
    g = decision.get("garch", {})
    q = decision.get("quantile", {})
    row = {
        "signal_id": signal_id,
        "timestamp": decision.get("timestamp"),
        "ticker": decision.get("ticker"),
        "current_price": decision.get("feature_row", {}).get("current_price")
            if "feature_row" in decision else sizing.get("entry_price"),
        "prediction_horizon": decision.get("prediction_horizon", "24h"),
        "xgb_downside_probability": decision.get("p_downside"),
        "adjusted_downside_probability": decision.get("adjusted_p_downside"),
        "xgb_confidence": abs(float(decision.get("adjusted_p_downside", 0.5)) - 0.5) * 2.0,
        "garch_volatility_forecast": g.get("forecast_volatility"),
        "garch_volatility_regime": g.get("volatility_regime"),
        "var_estimate": g.get("var_estimate"),
        "expected_shortfall_estimate": g.get("expected_shortfall_estimate"),
        "quantile_5pct_return": q.get("predicted_5pct_return"),
        "quantile_10pct_return": q.get("predicted_10pct_return"),
        "model_agreement_score": decision.get("model_agreement_score"),
        "data_confidence_score": decision.get("data_confidence_score"),
        "base_downside_risk_score": decision.get("base_downside_risk_score"),
        "adjusted_downside_risk_score": decision.get("adjusted_downside_risk_score"),
        "uncertainty_penalty": decision.get("uncertainty_penalty"),
        "price_volume_confirmation": decision.get("price_volume_confirmation"),
        "news_shift_logodds": decision.get("news_shift_logodds"),
        "expected_edge_bps": decision.get("expected_edge_bps"),
        "decision": decision.get("decision"),
        "recommended_position_size": sizing.get("position_size"),
        "position_notional": sizing.get("position_notional"),
        "stop_loss": sizing.get("stop_loss"),
        "take_profit": sizing.get("take_profit"),
        "top_3_signal_drivers": str(decision.get("top_drivers", [])[:3]),
        "news_catalysts": str(decision.get("news_catalysts", {})),
        "market_regime": decision.get("market_regime"),
        "actual_return_12h": np.nan,
        "actual_return_24h": np.nan,
        "prediction_correct": np.nan,
        "paper_trade_result": None,
        "post_trade_attribution": None,
        "created_at": now,
        "updated_at": now,
    }
    df = load_ledger()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _save_ledger(df)
    log.info("Logged prediction %s: %s %s @ %s.", signal_id, row["decision"],
             row["ticker"], row["timestamp"])
    return signal_id


def update_actual_outcome(signal_id: str, prices: pd.DataFrame, horizon_bars: int = 1) -> dict[str, Any]:
    """Fill realised next-session return + correctness for a matured signal.

    Returns the updated fields, or {} if the signal/outcome isn't available yet.
    """
    df = load_ledger()
    mask = df["signal_id"] == signal_id
    if not mask.any():
        return {}
    rowi = df[mask].index[0]
    ts = to_utc(pd.Timestamp(df.loc[rowi, "timestamp"]))

    p = prices.sort_index()
    idx = p.index
    after = idx[idx > ts]
    if len(after) < horizon_bars:
        return {}   # outcome not available yet

    entry_close = float(p.loc[idx[idx <= ts]]["close"].iloc[-1]) if (idx <= ts).any() else np.nan
    future_close = float(p.loc[after[horizon_bars - 1]]["close"])
    actual_return = (future_close - entry_close) / entry_close if entry_close else np.nan

    decision = df.loc[rowi, "decision"]
    # "Correct": a SHORT/WATCH downside call is right if the stock fell;
    # a NO TRADE is right if no meaningful downside occurred.
    if decision in ("SHORT", "WATCH"):
        correct = bool(actual_return < 0)
    else:
        correct = bool(actual_return >= -0.005)

    # Cast to object first so a bool can be stored without a float-dtype clash (pandas 2.x).
    df["prediction_correct"] = df["prediction_correct"].astype(object)
    df.loc[rowi, "actual_return_24h"] = actual_return
    df.loc[rowi, "prediction_correct"] = correct
    df.loc[rowi, "updated_at"] = now_utc().isoformat()
    _save_ledger(df)
    return {"actual_return_24h": actual_return, "prediction_correct": correct}
