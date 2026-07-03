"""Paper-trading engine (spec §16). Simulates trades — NO real money.

Given a signal, its sizing, and forward price bars, simulate a short entry/exit
and compute P&L. Used both live (entry now, outcome pending) and in the
out-of-sample paper backtest (entry + realized exit on historical bars).
"""
from __future__ import annotations

import uuid
from typing import Any

import pandas as pd

from src.trading.execution_rules import resolve_short_exit
from src.utils.logging_utils import get_logger
from src.utils.timestamp_utils import now_utc

log = get_logger("trading.paper")


def simulate_short_trade(
    signal_id: str,
    ticker: str,
    entry_time: pd.Timestamp,
    sizing: dict[str, Any],
    forward_bars: pd.DataFrame,
    horizon_bars: int = 1,
    market_regime: str | None = None,
) -> dict[str, Any]:
    """Simulate a short paper trade and return a completed trade record.

    ``forward_bars`` are the OHLC bars AFTER entry (already sliced). Only the
    first ``horizon_bars`` are considered before horizon expiry.
    """
    entry_price = float(sizing["entry_price"])
    shares = float(sizing["position_size"])
    window = forward_bars.iloc[:horizon_bars]
    exit_info = resolve_short_exit(entry_time, sizing["stop_loss"], sizing["take_profit"], window)

    exit_price = exit_info["exit_price"]
    # Short P&L: profit when price falls.
    return_pct = (entry_price - exit_price) / entry_price if entry_price else float("nan")
    pnl = shares * (entry_price - exit_price)
    result = "win" if pnl > 0 else ("loss" if pnl < 0 else "flat")

    return {
        "trade_id": uuid.uuid4().hex[:12],
        "signal_id": signal_id,
        "ticker": ticker.upper(),
        "side": "short",
        "entry_time": str(entry_time),
        "entry_price": entry_price,
        "position_size": shares,
        "position_notional": float(sizing.get("position_notional", 0.0)),
        "stop_loss": float(sizing["stop_loss"]),
        "take_profit": float(sizing["take_profit"]),
        "exit_time": str(exit_info["exit_time"]),
        "exit_price": exit_price,
        "exit_reason": exit_info["exit_reason"],
        "return_pct": float(return_pct),
        "pnl": float(pnl),
        "result": result,
        "market_regime": market_regime,
        "created_at": now_utc().isoformat(),
        "updated_at": now_utc().isoformat(),
    }
