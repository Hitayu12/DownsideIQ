"""Price provider (yfinance) — PRIMARY data source.

Hard rule (spec §6): if price data can't be fetched/validated, signal generation
is BLOCKED via ``DataQualityError`` — never a silent or fake fallback. Includes a
staleness guard so a stale last bar also blocks.
"""
from __future__ import annotations

from datetime import timedelta

import pandas as pd

from src.core.errors import DataQualityError
from src.core.logging import get_logger
from src.core.time import ensure_tz_aware, now_utc
from src.providers.base import call_with_retry, provider_cfg

log = get_logger("providers.prices")

_INTRADAY_MAX_DAYS = {"1m": 7, "5m": 59, "15m": 59, "30m": 59, "60m": 729, "1h": 729}
_BAR_TO_YF = {"1d": "1d", "1h": "1h", "15m": "15m", "5m": "5m", "1wk": "1wk"}


def _yf_interval(bar_size: str) -> str:
    if bar_size not in _BAR_TO_YF:
        raise ValueError(f"Unsupported bar_size '{bar_size}'. Known: {sorted(_BAR_TO_YF)}")
    return _BAR_TO_YF[bar_size]


def fetch_ohlcv(
    ticker: str,
    bar_size: str | None = None,
    lookback_days: int | None = None,
    *,
    require: bool = False,
) -> pd.DataFrame:
    """Fetch OHLCV bars. Returns a tz-aware UTC DataFrame.

    ``require=True`` raises ``DataQualityError`` on empty/failed data (use for the
    primary ticker, whose absence must block signals). ``require=False`` returns
    an empty frame on failure (use for context assets, which degrade gracefully).
    """
    cfg = provider_cfg("price")
    bar_size = bar_size or cfg.get("bar_size", "1d")
    lookback_days = lookback_days or int(cfg.get("history_lookback_days", 1500))
    interval = _yf_interval(bar_size)
    if interval in _INTRADAY_MAX_DAYS:
        lookback_days = min(lookback_days, _INTRADAY_MAX_DAYS[interval])

    start = (pd.Timestamp(now_utc()) - pd.Timedelta(days=lookback_days)).tz_localize(None)

    def _download() -> pd.DataFrame:
        import yfinance as yf

        return yf.download(ticker, start=start, interval=interval,
                           auto_adjust=False, progress=False, threads=False)

    try:
        df = call_with_retry(_download, provider="yfinance",
                             retries=int(cfg.get("retries", 2)),
                             backoff=float(cfg.get("retry_backoff_seconds", 2)))
    except Exception as exc:
        if require:
            raise DataQualityError(f"Price fetch failed for {ticker}: {exc}") from exc
        log.warning("price_fetch_failed", ticker=ticker, error=str(exc)[:160])
        return pd.DataFrame()

    if df is None or df.empty:
        if require:
            raise DataQualityError(f"No price data returned for {ticker} ({interval}).")
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                            "Close": "close", "Adj Close": "adj_close", "Volume": "volume"})
    keep = [c for c in ["open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]
    df = df[keep].copy()
    df.index = ensure_tz_aware(df.index)
    df.index.name = "timestamp"
    df = df[~df.index.duplicated(keep="last")].sort_index()

    if require:
        _validate_quality(df, ticker, bar_size, cfg)
    log.info("price_fetched", ticker=ticker, bars=len(df), interval=interval,
             last=str(df.index.max()))
    return df


def _validate_quality(df: pd.DataFrame, ticker: str, bar_size: str, cfg: dict) -> None:
    """Block on null closes, future timestamps, or stale last bar."""
    if df["close"].isna().any():
        raise DataQualityError(f"{ticker}: null close prices present.")
    last = df.index.max()
    if last > now_utc():
        raise DataQualityError(f"{ticker}: future timestamp in price data ({last}).")
    if bar_size == "1d":
        max_stale = int(cfg.get("max_staleness_days", 5))
        age_days = (now_utc() - last).days
        if age_days > max_stale:
            raise DataQualityError(
                f"{ticker}: price data stale (last bar {last}, {age_days}d > {max_stale}d)."
            )


def to_bar_rows(df: pd.DataFrame) -> list[dict]:
    """Convert an OHLCV frame to row dicts for ``repositories.upsert_price_bars``."""
    rows = []
    for ts, r in df.iterrows():
        rows.append({
            "ts": ts.to_pydatetime(),
            "open": float(r["open"]), "high": float(r["high"]), "low": float(r["low"]),
            "close": float(r["close"]),
            "adj_close": float(r["adj_close"]) if "adj_close" in r and pd.notna(r["adj_close"]) else None,
            "volume": float(r["volume"]) if "volume" in r and pd.notna(r["volume"]) else None,
        })
    return rows
