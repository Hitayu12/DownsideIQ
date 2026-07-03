"""Trade exit rules (spec §14.3/§14.4).

Given a short entry, its stop/take-profit, and the subsequent OHLC bars within
the prediction horizon, determine the exit price/time/reason. Conservative
tie-break: if a single bar's range touches BOTH stop and target, assume the
stop fired first (worst case for the trade).
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def resolve_short_exit(
    entry_time: pd.Timestamp,
    stop_loss: float,
    take_profit: float,
    forward_bars: pd.DataFrame,
) -> dict[str, Any]:
    """Resolve a SHORT exit over ``forward_bars`` (OHLC, indexed by time).

    Returns exit_time, exit_price, exit_reason. If neither stop nor target is
    hit within the horizon, exit at the last bar's close (horizon expiry).
    """
    if forward_bars.empty:
        return {"exit_time": entry_time, "exit_price": float("nan"), "exit_reason": "no_data"}

    for ts, bar in forward_bars.iterrows():
        high = float(bar["high"])
        low = float(bar["low"])
        # Short stop is ABOVE entry (adverse). Check stop before target (worst case).
        if high >= stop_loss:
            return {"exit_time": ts, "exit_price": float(stop_loss), "exit_reason": "stop_loss"}
        if low <= take_profit:
            return {"exit_time": ts, "exit_price": float(take_profit), "exit_reason": "take_profit"}

    last_ts = forward_bars.index[-1]
    return {
        "exit_time": last_ts,
        "exit_price": float(forward_bars.iloc[-1]["close"]),
        "exit_reason": "horizon_expiry",
    }
