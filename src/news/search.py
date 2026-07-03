"""Tavily news-search helper.

Wraps the Tavily client into a simple ``search_news`` call that returns a list
of normalised raw items. Returns ``[]`` (logged) when Tavily is unavailable, so
news agents degrade to "no events" rather than crashing.
"""
from __future__ import annotations

from typing import Any

from src.utils.api_clients import get_tavily_client
from src.utils.logging_utils import get_logger

log = get_logger("news.search")


def search_news(
    query: str,
    max_results: int = 5,
    days: int = 3,
    topic: str = "news",
) -> list[dict[str, Any]]:
    """Run one Tavily news query. Returns normalised items (possibly empty)."""
    client = get_tavily_client()
    if client is None:
        return []
    try:
        resp = client.search(
            query=query,
            topic=topic,
            max_results=max_results,
            days=days,
            search_depth="basic",
        )
    except Exception as exc:
        log.warning("Tavily search failed for %r: %s", query, exc)
        return []

    results = (resp or {}).get("results", []) if isinstance(resp, dict) else []
    items: list[dict[str, Any]] = []
    for r in results:
        items.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", "") or r.get("snippet", ""),
            "score": r.get("score"),
            "published_date": r.get("published_date") or r.get("published_at"),
            "query": query,
        })
    return items


def dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop duplicate items by URL (keeps first / highest-ranked)."""
    seen: set[str] = set()
    out = []
    for it in items:
        url = it.get("url", "")
        key = url or it.get("title", "")
        if key and key not in seen:
            seen.add(key)
            out.append(it)
    return out
