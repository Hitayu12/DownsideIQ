"""News scorer dispatcher.

Chooses the configured scorer (Gemini by default) and transparently falls back
to the heuristic scorer when the LLM is unavailable or fails. Recency is always
computed deterministically from timestamps (the LLM can't know "now"), so it is
recomputed here regardless of which scorer produced the row.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from src.news import gemini_scorer, heuristic_scorer
from src.news.schema import NewsScore
from src.utils.config_loader import get_settings, has_key
from src.utils.logging_utils import get_logger
from src.utils.timestamp_utils import now_utc

log = get_logger("news.scorer")


def _provider() -> str:
    pref = str(get_settings().get("news_scorer", "gemini")).lower()
    if pref == "gemini" and has_key("GEMINI_API_KEY"):
        return "gemini"
    return "heuristic"


def score_item(
    item: dict[str, Any],
    ticker: str,
    company: str | None = None,
    as_of: datetime | None = None,
    allow_llm: bool = True,
) -> NewsScore:
    """Score one news item, with automatic Gemini→heuristic fallback.

    ``allow_llm=False`` forces the heuristic scorer (used for low-priority items
    beyond the per-run LLM budget).
    """
    as_of = as_of or now_utc()
    score: NewsScore | None = None
    if allow_llm and _provider() == "gemini":
        score = gemini_scorer.score_item(item, ticker, company, as_of)
    if score is None:
        score = heuristic_scorer.score_item(item, ticker, company, as_of)

    # Recency is deterministic from timestamps — recompute it for consistency.
    recency, pub_iso = heuristic_scorer._recency(
        item.get("published_date") or item.get("published_at") or score.published_at, as_of
    )
    score.recency_score = recency
    if pub_iso and not score.published_at:
        score.published_at = pub_iso
    return score.normalized()


def _priority(item: dict[str, Any], as_of: datetime) -> float:
    """Cheap ranking signal: prefer relevant + recent items for LLM scoring."""
    relevance = item.get("score")
    relevance = float(relevance) if isinstance(relevance, (int, float)) else 0.3
    recency, _ = heuristic_scorer._recency(
        item.get("published_date") or item.get("published_at"), as_of
    )
    return 0.6 * relevance + 0.4 * recency


def score_items(
    items: list[dict[str, Any]],
    ticker: str,
    company: str | None = None,
    as_of: datetime | None = None,
) -> list[NewsScore]:
    """Score a batch of news items.

    Only the top-N most relevant/recent items are LLM-scored (cost control via
    ``news_scorer_max_llm_items``); the remainder use the heuristic scorer. The
    LLM circuit breaker also applies, so a quota failure mid-batch degrades the
    rest to heuristic automatically.
    """
    as_of = as_of or now_utc()
    max_llm = int(get_settings().get("news_scorer_max_llm_items", 20))
    use_llm = _provider() == "gemini"

    # Rank item indices by priority; the top ``max_llm`` are eligible for the LLM.
    order = sorted(range(len(items)), key=lambda i: _priority(items[i], as_of), reverse=True)
    llm_eligible = set(order[:max_llm]) if use_llm else set()

    out: list[NewsScore] = []
    for i, it in enumerate(items):
        out.append(score_item(it, ticker, company, as_of, allow_llm=(i in llm_eligible)))

    if out:
        from collections import Counter
        counts = Counter(s.scorer for s in out)
        log.info("Scored %d news items for %s (%s).", len(out), ticker,
                 ", ".join(f"{k}={v}" for k, v in counts.items()))
    return out
