"""Phase 3 feature-correctness tests (offline, deterministic)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.fundamental_features import compute_fundamental_features
from src.features.news_features import aggregate_company_news, aggregate_macro_news
from src.features.price_features import compute_price_features
from src.utils.timestamp_utils import to_utc


def _prices():
    idx = pd.date_range("2024-01-01", periods=60, freq="B", tz="UTC")
    close = pd.Series(np.linspace(100, 120, 60), index=idx)
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "adj_close": close, "volume": 1_000_000.0},
        index=idx,
    )


def test_return_computation_matches_manual():
    p = _prices()
    f = compute_price_features(p)
    manual_1b = p["close"].pct_change(1)
    pd.testing.assert_series_equal(
        f["return_1b"], manual_1b, check_names=False
    )


def test_first_row_returns_are_nan():
    f = compute_price_features(_prices())
    assert np.isnan(f["return_1b"].iloc[0])     # no prior bar -> NaN, not 0


def test_company_news_bearish_raises_risk():
    bearish = [{
        "expected_direction": "bearish", "sentiment_score": -0.8,
        "expected_impact_score": 0.9, "relevance_score": 0.9,
        "credibility_score": 0.9, "recency_score": 0.9,
        "company_specificity_score": 0.8, "confidence_score": 0.8,
    }]
    agg = aggregate_company_news(bearish)
    assert agg["company_news_risk_score"] > 0.3
    assert agg["negative_catalyst_score"] > agg["positive_catalyst_score"]


def test_company_news_bullish_lowers_risk():
    bullish = [{
        "expected_direction": "bullish", "sentiment_score": 0.8,
        "expected_impact_score": 0.9, "relevance_score": 0.9,
        "credibility_score": 0.9, "recency_score": 0.9,
        "company_specificity_score": 0.8, "confidence_score": 0.8,
    }]
    agg = aggregate_company_news(bullish)
    assert agg["company_news_risk_score"] < 0
    assert agg["positive_catalyst_score"] > agg["negative_catalyst_score"]


def test_empty_news_is_neutral():
    assert aggregate_company_news([])["company_news_risk_score"] == 0.0
    assert aggregate_macro_news([])["macro_risk_score"] == 0.0


def test_fundamentals_suppressed_before_release():
    """eps_surprise must be NaN if last reported earnings is not yet public."""
    as_of = to_utc("2026-02-01T00:00:00Z")
    fundamentals = {
        "overview": {"PERatio": "30"},
        "earnings": {"eps_surprise": "0.25"},
        "last_reported_earnings_date": "2026-02-05",  # AFTER as_of -> must suppress
        "earnings_date_distance_days": 4,
    }
    f = compute_fundamental_features(fundamentals, as_of=as_of)
    assert np.isnan(f["eps_surprise"])

    # Same data, but as_of after release -> now usable.
    f2 = compute_fundamental_features(fundamentals, as_of=to_utc("2026-02-06T00:00:00Z"))
    assert f2["eps_surprise"] == 0.25
