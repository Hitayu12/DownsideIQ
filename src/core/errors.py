"""Typed exception hierarchy + degraded-mode status (production error handling).

Rules (from the production brief):
  - yfinance failure        -> block signal generation (DataQualityError, hard)
  - Tavily failure          -> price-only mode, news_confidence = 0 (degraded)
  - Gemini failure          -> heuristic news scoring (degraded)
  - Alpha Vantage / Finnhub -> skip fundamentals (degraded)
No silent failures. Every degraded path sets an explicit, surfaced status.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DownsideIQError(Exception):
    """Base class for all DownsideIQ errors."""


class ConfigError(DownsideIQError):
    """Configuration / environment validation failure (fatal at startup)."""


class DataQualityError(DownsideIQError):
    """Data failed validation hard enough to BLOCK signal generation.

    Raised when price data is missing/stale/invalid, timestamps are in the
    future, or required feature fields are null. Must never be swallowed.
    """


class ProviderError(DownsideIQError):
    """An external provider call failed after retries."""

    def __init__(self, provider: str, message: str, *, transient: bool = False):
        self.provider = provider
        self.transient = transient
        super().__init__(f"[{provider}] {message}")


class ValidationError(DownsideIQError):
    """A domain/data-validation rule was violated."""


class ModelError(DownsideIQError):
    """Model load / inference failure."""


class DataQuality(str, Enum):
    """Overall data-quality status attached to every signal."""

    OK = "ok"                 # all primary + enrichment sources present
    DEGRADED = "degraded"     # price OK, some enrichment missing
    BLOCKED = "blocked"       # price data unusable -> no signal


@dataclass
class DegradedMode:
    """Tracks which optional capabilities are degraded for this request.

    Surfaced into the signal governance record so the dashboard/API can show
    *why* confidence is reduced — never a silent downgrade.
    """

    price_ok: bool = True
    news_available: bool = True
    news_scorer: str = "gemini"        # gemini | heuristic | none
    fundamentals_available: bool = True
    reasons: list[str] = field(default_factory=list)

    def degrade(self, capability: str, reason: str) -> None:
        self.reasons.append(f"{capability}: {reason}")
        if capability == "news":
            self.news_available = False
        elif capability == "news_scorer":
            self.news_scorer = "heuristic"
        elif capability == "fundamentals":
            self.fundamentals_available = False
        elif capability == "price":
            self.price_ok = False

    @property
    def status(self) -> DataQuality:
        if not self.price_ok:
            return DataQuality.BLOCKED
        if self.reasons:
            return DataQuality.DEGRADED
        return DataQuality.OK

    @property
    def news_confidence_multiplier(self) -> float:
        """0 when news unavailable, 0.7 on heuristic fallback, 1.0 when full."""
        if not self.news_available:
            return 0.0
        return 0.7 if self.news_scorer == "heuristic" else 1.0
