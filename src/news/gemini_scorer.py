"""Gemini-based structured news scorer (google-genai SDK).

Produces the same NewsScore schema as the heuristic scorer, but with LLM
judgement. Returns ``None`` on any failure (no key, network error, bad JSON) so
the dispatcher transparently falls back to the heuristic scorer.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from src.news.schema import LLM_OUTPUT_INSTRUCTIONS, NewsScore, _scope_for
from src.utils.api_clients import get_gemini_client
from src.utils.config_loader import get_env
from src.utils.logging_utils import get_logger
from src.utils.timestamp_utils import now_utc

log = get_logger("news.gemini")

_DEFAULT_MODEL = "gemini-2.5-flash"

# Circuit breaker: once the API returns a hard quota/permission/not-found error,
# stop calling Gemini for the rest of the process and use the heuristic scorer.
# Avoids hammering a dead quota hundreds of times (and the log spam that causes).
_DISABLED = False
_DISABLED_REASON = ""
_FATAL_MARKERS = ("RESOURCE_EXHAUSTED", "PERMISSION_DENIED", "NOT_FOUND",
                   "API_KEY_INVALID", "429", "403", "404")


def is_disabled() -> bool:
    return _DISABLED


def _trip_breaker(exc: Exception) -> None:
    global _DISABLED, _DISABLED_REASON
    _DISABLED = True
    _DISABLED_REASON = str(exc)[:160]
    log.warning("Gemini disabled for this run (%s). Using heuristic scorer.", _DISABLED_REASON)


def _is_fatal(exc: Exception) -> bool:
    msg = str(exc)
    return any(marker in msg for marker in _FATAL_MARKERS)


def _build_prompt(item: dict[str, Any], ticker: str, company: str | None) -> str:
    title = item.get("title", "")
    content = (item.get("content") or item.get("snippet") or "")[:1500]
    url = item.get("url", "")
    name = company or ticker
    return (
        f"You are a sell-side risk analyst scoring a news item for its 12-24 hour "
        f"DOWNSIDE risk impact on {name} (ticker {ticker}).\n\n"
        f"HEADLINE: {title}\n"
        f"SOURCE URL: {url}\n"
        f"EXCERPT: {content}\n\n"
        f"{LLM_OUTPUT_INSTRUCTIONS}"
    )


def _parse(raw: str) -> dict[str, Any] | None:
    raw = raw.strip()
    # Strip ```json fences if the model added them.
    if raw.startswith("```"):
        raw = raw.split("```")[1] if "```" in raw[3:] else raw.strip("`")
        raw = raw[len("json"):].strip() if raw.lower().startswith("json") else raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def score_item(
    item: dict[str, Any],
    ticker: str,
    company: str | None = None,
    as_of: datetime | None = None,
) -> NewsScore | None:
    """Score one news item with Gemini. Returns None on any failure."""
    if _DISABLED:
        return None
    client = get_gemini_client()
    if client is None:
        return None
    as_of = as_of or now_utc()
    model = get_env("GEMINI_MODEL", _DEFAULT_MODEL)
    try:
        resp = client.models.generate_content(
            model=model,
            contents=_build_prompt(item, ticker, company),
            config={"response_mime_type": "application/json", "temperature": 0.0},
        )
        data = _parse(resp.text or "")
    except Exception as exc:
        # Hard quota/permission errors trip the breaker so we stop retrying.
        if _is_fatal(exc):
            _trip_breaker(exc)
        else:
            log.warning("Gemini scoring failed (%s); falling back to heuristic.", str(exc)[:160])
        return None

    if not data:
        log.warning("Gemini returned unparseable output; falling back to heuristic.")
        return None

    etype = data.get("event_type", "unknown")
    return NewsScore(
        event_type=etype,
        scope=data.get("scope") or _scope_for(etype),
        sentiment_score=data.get("sentiment_score", 0.0),
        relevance_score=data.get("relevance_score", 0.0),
        credibility_score=data.get("credibility_score", 0.5),
        recency_score=data.get("recency_score", 0.5),
        company_specificity_score=data.get("company_specificity_score", 0.0),
        expected_direction=data.get("expected_direction", "neutral"),
        expected_impact_score=data.get("expected_impact_score", 0.0),
        confidence_score=data.get("confidence_score", 0.0),
        raw_summary=data.get("raw_summary", item.get("title", "")),
        source_url=item.get("url", ""),
        published_at=str(item.get("published_date") or item.get("published_at") or "") or None,
        scored_at=as_of.isoformat(),
        scorer="gemini",
    ).normalized()
