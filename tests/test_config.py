"""Phase 1 smoke tests: configs load, settings are sane, CLI status runs."""
from __future__ import annotations

import pytest

from src.utils.config_loader import (
    api_key_status,
    get_event_playbook,
    get_risk_limits,
    get_settings,
    get_ticker_config,
    get_tickers,
)


def test_settings_load():
    s = get_settings()
    assert s["default_ticker"] == "NVDA"
    assert s["bar_size"] in {"1d", "1h", "15m"}
    assert "target" in s and "type" in s["target"]


def test_risk_limits_sane():
    r = get_risk_limits()
    # Live trading must be off by default — this is a hard safety invariant.
    assert r["live_trading_enabled"] is False
    assert r["paper_trading_mode"] is True
    assert 0 < r["max_risk_per_trade"] < r["max_daily_loss"] < r["max_weekly_drawdown"]
    assert 0 < r["minimum_downside_probability"] <= 1


def test_default_ticker_configured():
    cfg = get_ticker_config("nvda")  # case-insensitive
    assert "SPY" in cfg["market_etfs"]
    assert cfg["peers"], "NVDA should have peer context assets"


def test_unknown_ticker_raises():
    from src.core.errors import ConfigError

    with pytest.raises(ConfigError):
        get_ticker_config("DOES_NOT_EXIST")


def test_event_playbook_has_fallback_and_bias_signs():
    pb = get_event_playbook()["events"]
    assert "unknown" in pb, "playbook needs an 'unknown' fallback event"
    # Bearish events raise downside risk (bias > 0); bullish lower it (< 0).
    assert pb["guidance_cut"]["bias"] > 0
    assert pb["earnings_beat"]["bias"] < 0


def test_api_key_status_never_raises():
    status = api_key_status()
    assert set(status) >= {"tavily", "gemini", "alpha_vantage", "finnhub"}
    assert all(isinstance(v, bool) for v in status.values())


def test_cli_status_runs():
    from main import main

    assert main(["--status"]) == 0
