"""Layer 3 — provider interface + failure-mode tests (mocks only).

These assert the graceful-degradation contracts from spec §6. No model metric
is derived from mock data — these only exercise interfaces and failure paths.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.core.errors import DataQualityError
from src.core.time import now_utc
from src.providers import (
    alpha_vantage,
    finnhub_provider,
    gemini_provider,
    prices,
    tavily_provider,
)


def _ohlcv(index):
    return pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0,
                         "Adj Close": 1.0, "Volume": 100.0}, index=index)


# --------------------------- prices (HARD: failure blocks) --------------------------- #
def test_price_failure_blocks_signal(monkeypatch):
    import yfinance
    monkeypatch.setattr(yfinance, "download", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("network")))
    with pytest.raises(DataQualityError):
        prices.fetch_ohlcv("NVDA", require=True)


def test_price_empty_blocks_when_required(monkeypatch):
    import yfinance
    monkeypatch.setattr(yfinance, "download", lambda *a, **k: pd.DataFrame())
    with pytest.raises(DataQualityError):
        prices.fetch_ohlcv("NVDA", require=True)


def test_price_context_asset_degrades_quietly(monkeypatch):
    import yfinance
    monkeypatch.setattr(yfinance, "download", lambda *a, **k: pd.DataFrame())
    df = prices.fetch_ohlcv("SPY", require=False)   # context asset -> empty, no raise
    assert df.empty


def test_price_future_timestamp_blocked(monkeypatch):
    import yfinance
    future = pd.date_range(pd.Timestamp(now_utc()).normalize() + pd.Timedelta(days=2), periods=3, freq="D", tz="UTC")
    monkeypatch.setattr(yfinance, "download", lambda *a, **k: _ohlcv(future))
    with pytest.raises(DataQualityError):
        prices.fetch_ohlcv("NVDA", require=True)


def test_price_valid_recent_passes(monkeypatch):
    import yfinance
    idx = pd.date_range(pd.Timestamp(now_utc()).normalize() - pd.Timedelta(days=4), periods=5, freq="D", tz="UTC")
    monkeypatch.setattr(yfinance, "download", lambda *a, **k: _ohlcv(idx))
    df = prices.fetch_ohlcv("NVDA", require=True)
    assert list(df.columns) == ["open", "high", "low", "close", "adj_close", "volume"]
    rows = prices.to_bar_rows(df)
    assert len(rows) == 5 and "ts" in rows[0]


# --------------------------- tavily (degrade to []) --------------------------- #
def test_tavily_unavailable_returns_empty(monkeypatch):
    monkeypatch.setattr(tavily_provider, "_client", lambda: None)
    assert tavily_provider.search_news("anything") == []


def test_tavily_search_error_returns_empty(monkeypatch):
    class Boom:
        def search(self, **k):
            raise RuntimeError("api down")
    monkeypatch.setattr(tavily_provider, "_client", lambda: Boom())
    assert tavily_provider.search_news("q") == []


# --------------------------- gemini (circuit breaker) --------------------------- #
def test_gemini_breaker_trips_on_quota(monkeypatch):
    gemini_provider.reset_breaker()

    class FakeModels:
        def generate_content(self, **k):
            raise RuntimeError("429 RESOURCE_EXHAUSTED quota")

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(gemini_provider, "_client", lambda: FakeClient())
    assert gemini_provider.generate_json("prompt") is None
    assert gemini_provider.is_disabled() is True
    # Once disabled, further calls short-circuit to None.
    assert gemini_provider.generate_json("prompt2") is None
    gemini_provider.reset_breaker()


# --------------------------- alpha vantage (skip fundamentals) --------------------------- #
def test_alpha_vantage_rate_limit_note_returns_none(monkeypatch):
    monkeypatch.setattr(alpha_vantage.env(), "alpha_vantage_api_key", "x", raising=False)

    class FakeResp:
        def raise_for_status(self): ...
        def json(self): return {"Note": "rate limit, slow down"}

    monkeypatch.setattr(alpha_vantage.requests, "get", lambda *a, **k: FakeResp())
    assert alpha_vantage.query({"function": "OVERVIEW", "symbol": "NVDA"}) is None


# --------------------------- finnhub (degrade) --------------------------- #
def test_finnhub_unavailable_returns_empty(monkeypatch):
    monkeypatch.setattr(finnhub_provider, "_client", lambda: None)
    assert finnhub_provider.company_news("NVDA") == []
    assert finnhub_provider.basic_financials("NVDA") == {}
    assert finnhub_provider.next_earnings_date("NVDA") is None
