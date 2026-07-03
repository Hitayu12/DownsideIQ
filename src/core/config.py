"""Configuration + environment management (canonical home).

- ``EnvSettings`` (pydantic-settings) loads secrets from ``.env`` with clearly
  separated REQUIRED vs OPTIONAL keys and validates them at startup.
- YAML loaders expose the five config files (settings, risk_limits,
  model_config, data_sources, thresholds) plus tickers + event playbook.

No secret is ever hardcoded. Missing OPTIONAL keys degrade gracefully; a missing
REQUIRED key (or live trading enabled) raises ``ConfigError`` at startup.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.core.errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

# Optional API keys — system degrades gracefully if any are missing.
OPTIONAL_KEYS = ("TAVILY_API_KEY", "GEMINI_API_KEY", "ALPHA_VANTAGE_API_KEY", "FINNHUB_API_KEY")
# No key is strictly REQUIRED: price data (yfinance) needs none, so the system
# can always run in price-only mode. Required infra (DB/redis) have safe defaults.
REQUIRED_KEYS: tuple[str, ...] = ()


class EnvSettings(BaseSettings):
    """Typed environment settings loaded from ``.env`` (+ real env vars)."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore",
        case_sensitive=False,
    )

    # --- Optional provider keys ---
    tavily_api_key: str | None = None
    gemini_api_key: str | None = None
    alpha_vantage_api_key: str | None = None
    finnhub_api_key: str | None = None

    # --- Provider config ---
    gemini_model: str = "gemini-2.5-flash"
    news_scorer: str = "gemini"

    # --- Infra ---
    database_url: str = "sqlite:///downsideiq.db"
    redis_url: str = "redis://localhost:6379/0"

    # --- Defaults ---
    default_ticker: str = "NVDA"
    default_horizon: str = "24h"
    default_bar_size: str = "1d"
    environment: str = "development"

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- Logging ---
    log_level: str = "INFO"
    log_dir: str = "logs"
    log_json: bool = True

    # --- Safety ---
    paper_trading_mode: bool = True
    live_trading_enabled: bool = False

    def key_status(self) -> dict[str, bool]:
        return {
            "tavily": bool(self.tavily_api_key),
            "gemini": bool(self.gemini_api_key),
            "alpha_vantage": bool(self.alpha_vantage_api_key),
            "finnhub": bool(self.finnhub_api_key),
        }


@lru_cache(maxsize=1)
def env() -> EnvSettings:
    return EnvSettings()


def validate_environment() -> dict[str, Any]:
    """Validate env at startup. Raises ConfigError on a fatal misconfig.

    Returns a status dict (present/missing keys, degraded capabilities) for
    logging. Never leaks secret values.
    """
    e = env()
    import os

    missing_required = [k for k in REQUIRED_KEYS if not os.getenv(k)]
    if missing_required:
        raise ConfigError(f"Missing REQUIRED environment variables: {missing_required}")
    if e.live_trading_enabled:
        raise ConfigError("LIVE_TRADING_ENABLED must be false (live trading is disabled by design).")

    status = e.key_status()
    return {
        "environment": e.environment,
        "required_present": True,
        "optional_present": [k for k, v in status.items() if v],
        "optional_missing": [k for k, v in status.items() if not v],
        "news_scorer": e.news_scorer,
        "database": e.database_url.split("://")[0],
        "paper_trading_mode": e.paper_trading_mode,
        "live_trading_enabled": e.live_trading_enabled,
    }


# --------------------------------------------------------------------------- #
# YAML config loaders
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def get_settings() -> dict[str, Any]:
    return _load_yaml("settings.yaml")


def get_risk_limits() -> dict[str, Any]:
    return _load_yaml("risk_limits.yaml")


def get_tickers() -> dict[str, Any]:
    return _load_yaml("tickers.yaml")


def get_event_playbook() -> dict[str, Any]:
    return _load_yaml("event_playbook.yaml")


def get_model_config() -> dict[str, Any]:
    return _load_yaml("model_config.yaml")


def get_data_sources() -> dict[str, Any]:
    return _load_yaml("data_sources.yaml")


def get_thresholds() -> dict[str, Any]:
    return _load_yaml("thresholds.yaml")


def get_ticker_config(ticker: str) -> dict[str, Any]:
    tickers = get_tickers().get("tickers", {})
    ticker = ticker.upper()
    if ticker not in tickers:
        raise ConfigError(f"Ticker '{ticker}' not in config/tickers.yaml. Known: {sorted(tickers)}")
    return tickers[ticker]


# --------------------------------------------------------------------------- #
# Back-compat helpers (used by the MVP modules during migration)
# --------------------------------------------------------------------------- #
def get_env(key: str, default: str | None = None) -> str | None:
    import os

    return os.getenv(key, default)


def has_key(key: str) -> bool:
    import os

    v = os.getenv(key)
    return bool(v and v.strip())


def api_key_status() -> dict[str, bool]:
    return env().key_status()


def project_root() -> Path:
    return PROJECT_ROOT


def data_dir() -> Path:
    return PROJECT_ROOT / "data"
