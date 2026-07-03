"""Tavily news-search provider.

Failure rule (spec §6): on any failure the system runs in price-only mode and
the caller sets ``news_confidence = 0`` — returns ``[]``, never raises.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from src.core.config import env
from src.core.logging import get_logger
from src.providers.base import call_with_retry, provider_cfg

log = get_logger("providers.tavily")


@lru_cache(maxsize=1)
def _client():
    e = env()
    if not e.tavily_api_key:
        log.warning("tavily_key_missing")
        return None
    try:
        from tavily import TavilyClient

        return TavilyClient(api_key=e.tavily_api_key)
    except Exception as exc:  # pragma: no cover
        log.warning("tavily_init_failed", error=str(exc)[:160])
        return None


def available() -> bool:
    return _client() is not None


def search_news(query: str, max_results: int | None = None, days: int | None = None) -> list[dict[str, Any]]:
    """Return normalised news items, or [] on unavailable/failure (degraded)."""
    client = _client()
    if client is None:
        return []
    cfg = provider_cfg("news")
    max_results = max_results or int(cfg.get("max_results_per_query", 4))
    days = days or int(cfg.get("lookback_days", 3))

    def _search():
        return client.search(query=query, topic="news", max_results=max_results,
                             days=days, search_depth="basic")

    try:
        resp = call_with_retry(_search, provider="tavily",
                               retries=int(cfg.get("retries", 2)),
                               backoff=float(cfg.get("retry_backoff_seconds", 2)))
    except Exception as exc:
        log.warning("tavily_search_failed", query=query, error=str(exc)[:160])
        return []

    results = (resp or {}).get("results", []) if isinstance(resp, dict) else []
    return [{
        "title": r.get("title", ""), "url": r.get("url", ""),
        "content": r.get("content", "") or r.get("snippet", ""),
        "score": r.get("score"),
        "published_date": r.get("published_date") or r.get("published_at"),
        "query": query,
    } for r in results]
