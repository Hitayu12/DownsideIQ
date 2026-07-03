"""Market Context Agent (Council 1, spec §8.5).

Collects OHLCV for the context assets that determine whether the target's risk
is company-specific, sector-driven, or broad-market-driven: market ETFs
(SPY/QQQ), sector ETFs (SMH/SOXX), peers, and a volatility proxy (^VIX).

Raw bars only; relative-strength / beta / correlation features are computed in
the Feature Engineering Layer (Phase 3).
"""
from __future__ import annotations

import pandas as pd

from src.utils.config_loader import get_ticker_config
from src.utils.data_loader import fetch_ohlcv, save_frame
from src.utils.logging_utils import get_logger

log = get_logger("agents.market_context")


def context_assets_for(ticker: str) -> list[str]:
    """Ordered, de-duplicated list of context assets for a ticker."""
    cfg = get_ticker_config(ticker)
    assets: list[str] = []
    for key in ("market_etfs", "sector_etfs", "peers"):
        assets.extend(cfg.get(key, []))
    vol_proxy = cfg.get("vol_proxy")
    if vol_proxy:
        assets.append(vol_proxy)
    seen: set[str] = set()
    ordered = []
    for a in assets:
        if a not in seen:
            seen.add(a)
            ordered.append(a)
    return ordered


def collect_market_context(
    ticker: str,
    bar_size: str | None = None,
    lookback_days: int | None = None,
    save: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for every context asset. Returns {symbol: DataFrame}.

    Missing/failed symbols are skipped (logged), never fatal.
    """
    ticker = ticker.upper()
    assets = context_assets_for(ticker)
    out: dict[str, pd.DataFrame] = {}
    for symbol in assets:
        df = fetch_ohlcv(symbol, bar_size=bar_size, lookback_days=lookback_days)
        if df.empty:
            log.warning("Context asset %s returned no data; skipping.", symbol)
            continue
        out[symbol] = df
        if save:
            # ^VIX -> safe filename
            safe = symbol.replace("^", "_")
            save_frame(df, "prices", safe)
    log.info("Collected market context for %s: %d/%d assets.", ticker, len(out), len(assets))
    return out
