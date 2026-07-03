"""Macro & Geopolitical News Agent (Council 1, spec §8.1).

Searches for broad-market / macro events (Fed, inflation, jobs, yields,
recession, geopolitics, sector rotation) via Tavily, scores each into the
structured NewsScore schema, and persists raw + scored payloads. Output feeds
the macro side of the live news overlay.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from src.news.schema import NewsScore
from src.news.scorer import score_items
from src.news.search import dedupe, search_news
from src.utils.data_loader import save_json
from src.utils.logging_utils import get_logger
from src.utils.timestamp_utils import now_utc

log = get_logger("agents.macro_news")

_MACRO_QUERIES = [
    "Federal Reserve interest rate decision",
    "US CPI inflation report",
    "US jobs report nonfarm payrolls",
    "Treasury yields move stock market",
    "US recession risk economy",
    "geopolitical risk markets tariffs",
    "semiconductor sector selloff",
    "stock market risk off selloff",
]


def collect_macro_news(
    ticker: str,
    max_results_per_query: int = 4,
    days: int = 3,
    as_of: datetime | None = None,
    save: bool = True,
) -> dict[str, Any]:
    """Collect + score macro news. Returns {'raw': [...], 'scored': [...]}.

    Always returns a dict; empty lists if Tavily is unavailable.
    """
    ticker = ticker.upper()
    as_of = as_of or now_utc()

    raw: list[dict[str, Any]] = []
    for q in _MACRO_QUERIES:
        raw.extend(search_news(q, max_results=max_results_per_query, days=days))
    raw = dedupe(raw)

    scored: list[NewsScore] = score_items(raw, ticker, company=None, as_of=as_of)
    scored_dicts = [s.to_dict() for s in scored]

    payload = {"ticker": ticker, "as_of": as_of.isoformat(), "raw": raw, "scored": scored_dicts}
    if save:
        save_json(payload, "macro", f"{ticker}_macro_news")
    log.info("Macro news for %s: %d raw -> %d scored.", ticker, len(raw), len(scored_dicts))
    return payload
