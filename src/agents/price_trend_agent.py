"""Historical Stock Trend Agent (Council 1, spec §8.3).

Collects OHLCV history for the target ticker. Does NOT compute features here —
that is the Feature Engineering Layer's job (Phase 3). This agent only gathers
and persists clean, timestamped price bars.
"""
from __future__ import annotations

import pandas as pd

from src.utils.data_loader import fetch_ohlcv, save_frame
from src.utils.logging_utils import get_logger

log = get_logger("agents.price")


def collect_prices(
    ticker: str,
    bar_size: str | None = None,
    lookback_days: int | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Fetch and (optionally) persist OHLCV bars for ``ticker``.

    Returns the price DataFrame (possibly empty if yfinance is unavailable).
    """
    ticker = ticker.upper()
    df = fetch_ohlcv(ticker, bar_size=bar_size, lookback_days=lookback_days)
    if df.empty:
        log.warning("No price data collected for %s.", ticker)
        return df
    if save:
        save_frame(df, "prices", ticker)
    return df
