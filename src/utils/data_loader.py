"""Data fetching + raw persistence for DownsideIQ.

Price data comes from yfinance (no key). Everything is normalised to a
tz-aware UTC index so leakage checks in ``timestamp_utils`` are meaningful.
Raw payloads are saved under ``data/raw/<category>/`` as parquet (frames) or
JSON (news/fundamentals) for reproducibility and the future news dataset.
"""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config_loader import data_dir, get_settings
from src.utils.logging_utils import get_logger
from src.utils.timestamp_utils import ensure_tz_aware, now_utc

log = get_logger("data_loader")

# yfinance interval -> max history it will serve (approx; intraday is limited).
_INTRADAY_MAX_DAYS = {"1m": 7, "5m": 59, "15m": 59, "30m": 59, "60m": 729, "1h": 729}
_BAR_TO_YF_INTERVAL = {"1d": "1d", "1h": "1h", "15m": "15m", "5m": "5m", "1wk": "1wk"}


def _yf_interval(bar_size: str) -> str:
    if bar_size not in _BAR_TO_YF_INTERVAL:
        raise ValueError(f"Unsupported bar_size '{bar_size}'. Known: {sorted(_BAR_TO_YF_INTERVAL)}")
    return _BAR_TO_YF_INTERVAL[bar_size]


def fetch_ohlcv(
    ticker: str,
    bar_size: str | None = None,
    lookback_days: int | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV bars for ``ticker`` from yfinance.

    Returns a DataFrame indexed by a tz-aware UTC DatetimeIndex with columns
    ``[open, high, low, close, adj_close, volume]``. Empty DataFrame on failure
    (logged) so callers can degrade gracefully. Intraday lookback is clamped to
    yfinance limits.
    """
    import yfinance as yf

    settings = get_settings()
    bar_size = bar_size or settings.get("bar_size", "1d")
    lookback_days = lookback_days or int(settings.get("history_lookback_days", 1500))
    interval = _yf_interval(bar_size)

    if interval in _INTRADAY_MAX_DAYS:
        capped = min(lookback_days, _INTRADAY_MAX_DAYS[interval])
        if capped < lookback_days:
            log.info("Clamping %s lookback %dd -> %dd (yfinance intraday limit).",
                     interval, lookback_days, capped)
        lookback_days = capped

    end_ts = pd.Timestamp(end).tz_convert("UTC") if end is not None and pd.Timestamp(end).tzinfo \
        else (pd.Timestamp(end) if end is not None else now_utc())
    start_ts = pd.Timestamp(end_ts) - timedelta(days=lookback_days)

    try:
        df = yf.download(
            ticker,
            start=start_ts.tz_localize(None) if start_ts.tzinfo else start_ts,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        log.warning("yfinance download failed for %s (%s): %s", ticker, interval, exc)
        return pd.DataFrame()

    if df is None or df.empty:
        log.warning("yfinance returned no data for %s (%s).", ticker, interval)
        return pd.DataFrame()

    # yfinance may return a MultiIndex column frame for single tickers; flatten.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
    })
    keep = [c for c in ["open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]
    df = df[keep].copy()
    df.index = ensure_tz_aware(df.index)            # naive daily index -> UTC
    df.index.name = "timestamp"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    log.info("Fetched %d %s bars for %s (%s -> %s).",
             len(df), interval, ticker, df.index.min(), df.index.max())
    return df


# ----------------------------------------------------------------------------
# Raw persistence
# ----------------------------------------------------------------------------
def _raw_path(category: str, name: str, ext: str) -> Path:
    d = data_dir() / "raw" / category
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.{ext}"


def save_frame(df: pd.DataFrame, category: str, name: str) -> Path:
    """Persist a DataFrame as parquet under data/raw/<category>/<name>.parquet."""
    path = _raw_path(category, name, "parquet")
    df.to_parquet(path)
    log.info("Saved %d rows -> %s", len(df), path)
    return path


def load_frame(category: str, name: str) -> pd.DataFrame | None:
    path = _raw_path(category, name, "parquet")
    if not path.exists():
        return None
    return pd.read_parquet(path)


def save_json(obj: Any, category: str, name: str) -> Path:
    """Persist a JSON-serialisable object under data/raw/<category>/<name>.json."""
    path = _raw_path(category, name, "json")
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    log.info("Saved JSON -> %s", path)
    return path


def load_json(category: str, name: str) -> Any | None:
    path = _raw_path(category, name, "json")
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
