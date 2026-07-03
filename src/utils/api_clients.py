"""Thin, fail-soft clients for DownsideIQ's external data providers.

Every accessor returns ``None`` (and logs a warning) when its API key is
missing or the SDK can't initialise, so the pipeline degrades gracefully
instead of crashing. Network/HTTP errors are the caller's responsibility to
catch, but these constructors never raise on a missing key.

Providers:
    - Tavily         : live news search          (TAVILY_API_KEY)
    - Gemini         : structured news scoring    (GEMINI_API_KEY)
    - Alpha Vantage  : fundamentals / earnings    (ALPHA_VANTAGE_API_KEY)  [HTTP]
    - Finnhub        : company news / financials  (FINNHUB_API_KEY)
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import requests

from src.utils.config_loader import get_env, has_key
from src.utils.logging_utils import get_logger

log = get_logger("api_clients")

ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"
_HTTP_TIMEOUT = 20


@lru_cache(maxsize=1)
def get_tavily_client():
    """Return a Tavily client, or None if unavailable."""
    if not has_key("TAVILY_API_KEY"):
        log.warning("TAVILY_API_KEY missing — news search disabled.")
        return None
    try:
        from tavily import TavilyClient

        return TavilyClient(api_key=get_env("TAVILY_API_KEY"))
    except Exception as exc:  # pragma: no cover - import/init guard
        log.warning("Tavily client init failed: %s", exc)
        return None


@lru_cache(maxsize=1)
def get_gemini_client():
    """Return a google-genai client, or None if unavailable."""
    if not has_key("GEMINI_API_KEY"):
        log.warning("GEMINI_API_KEY missing — LLM news scoring disabled (heuristic fallback).")
        return None
    try:
        from google import genai

        return genai.Client(api_key=get_env("GEMINI_API_KEY"))
    except Exception as exc:  # pragma: no cover - import/init guard
        log.warning("Gemini client init failed: %s", exc)
        return None


@lru_cache(maxsize=1)
def get_finnhub_client():
    """Return a Finnhub client, or None if unavailable."""
    if not has_key("FINNHUB_API_KEY"):
        log.warning("FINNHUB_API_KEY missing — Finnhub data disabled.")
        return None
    try:
        import finnhub

        return finnhub.Client(api_key=get_env("FINNHUB_API_KEY"))
    except Exception as exc:  # pragma: no cover - import/init guard
        log.warning("Finnhub client init failed: %s", exc)
        return None


def alpha_vantage_get(params: dict[str, Any]) -> dict[str, Any] | None:
    """Call the Alpha Vantage REST API. Returns parsed JSON, or None on failure.

    Alpha Vantage signals rate limits / errors inside a 200 response (keys
    ``Note`` / ``Information`` / ``Error Message``); we detect and surface those.
    """
    if not has_key("ALPHA_VANTAGE_API_KEY"):
        log.warning("ALPHA_VANTAGE_API_KEY missing — Alpha Vantage data disabled.")
        return None
    q = {**params, "apikey": get_env("ALPHA_VANTAGE_API_KEY")}
    try:
        resp = requests.get(ALPHA_VANTAGE_BASE, params=q, timeout=_HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.warning("Alpha Vantage request failed (%s): %s", params.get("function"), exc)
        return None

    for flag in ("Note", "Information", "Error Message"):
        if flag in data:
            log.warning("Alpha Vantage %s: %s", flag, str(data[flag])[:160])
            # Rate-limit/usage notes mean no usable payload.
            if flag in ("Note", "Error Message"):
                return None
    return data


def provider_availability() -> dict[str, bool]:
    """Lightweight availability snapshot (key presence only, no network calls)."""
    return {
        "tavily": has_key("TAVILY_API_KEY"),
        "gemini": has_key("GEMINI_API_KEY"),
        "alpha_vantage": has_key("ALPHA_VANTAGE_API_KEY"),
        "finnhub": has_key("FINNHUB_API_KEY"),
    }
