"""Layer 4 — feature service data-validation contract (offline, no DB/network)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.errors import DataQualityError
from src.core.time import to_utc
from src.domain.features import IngestionResult
from src.services.feature_service import FeatureService


def _prices(end="2026-05-29", n=120):
    idx = pd.date_range(end=end, periods=n, freq="B", tz="UTC")
    close = pd.Series(np.linspace(100, 120, n), index=idx)
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "adj_close": close, "volume": 1e6}, index=idx)


def _ingestion(prices, as_of="2026-05-30"):
    return IngestionResult(ticker="NVDA", as_of=to_utc(as_of).to_pydatetime(),
                           bar_size="1d", prices=prices)


def test_snapshot_builds_and_scores_confidence():
    snap = FeatureService().build_snapshot(_ingestion(_prices()), persist=False)
    assert 0.0 <= snap.data_confidence_score <= 1.0
    assert "current_price" in snap.features
    assert snap.ticker == "NVDA"


def test_empty_prices_blocks():
    with pytest.raises(DataQualityError):
        FeatureService().build_snapshot(_ingestion(pd.DataFrame()), persist=False)


def test_future_feature_timestamp_blocks():
    # Last bar dated after as_of must raise (no look-ahead at decision time).
    prices = _prices(end="2026-06-10")
    ing = _ingestion(prices, as_of="2026-05-29")
    with pytest.raises(DataQualityError):
        FeatureService().build_snapshot(ing, persist=False)


def test_degraded_news_lowers_confidence():
    base = FeatureService().build_snapshot(_ingestion(_prices()), persist=False)
    ing = _ingestion(_prices())
    ing.degraded.degrade("news", "tavily unavailable")
    degraded = FeatureService().build_snapshot(ing, persist=False)
    assert degraded.data_confidence_score < base.data_confidence_score
