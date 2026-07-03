"""Time + look-ahead-bias utilities (canonical home; ported from utils).

The backbone of DownsideIQ's leakage discipline: every feature must use only
data available strictly before the prediction timestamp.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas as pd

UTC = timezone.utc


@lru_cache(maxsize=1)
def market_tz() -> ZoneInfo:
    # Imported lazily to avoid a config<->time import cycle at module load.
    from src.core.config import get_settings

    return ZoneInfo(get_settings().get("timezone", "America/New_York"))


def now_utc() -> datetime:
    return datetime.now(tz=UTC)


def now_market() -> datetime:
    return datetime.now(tz=market_tz())


def to_utc(ts: datetime | pd.Timestamp | str) -> pd.Timestamp:
    """Coerce to a tz-aware UTC Timestamp (naive assumed UTC, conservative)."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize(UTC)
    return t.tz_convert(UTC)


def ensure_tz_aware(index: pd.Index, assume_tz=UTC) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize(assume_tz)
    return idx


class LookAheadError(AssertionError):
    """Raised when data at/after the prediction timestamp would be used."""


def assert_no_future_data(frame: pd.DataFrame, as_of, time_col: str | None = None) -> None:
    """Raise LookAheadError if any row is dated at/after ``as_of``."""
    cutoff = to_utc(as_of)
    if time_col is not None:
        times = pd.to_datetime(frame[time_col], utc=True)
    else:
        times = ensure_tz_aware(frame.index).tz_convert(UTC)
    if len(times) == 0:
        return
    if times.max() >= cutoff:
        raise LookAheadError(
            f"Look-ahead: data timestamp {times.max()} >= prediction cutoff {cutoff}."
        )


def filter_strictly_before(frame: pd.DataFrame, as_of, time_col: str | None = None) -> pd.DataFrame:
    cutoff = to_utc(as_of)
    if time_col is not None:
        return frame.loc[pd.to_datetime(frame[time_col], utc=True) < cutoff].copy()
    idx = ensure_tz_aware(frame.index).tz_convert(UTC)
    return frame.loc[idx < cutoff].copy()
