"""Structured news event schema (shared by heuristic + Gemini scorers).

Every scorer — heuristic or LLM — must emit this exact schema so the live news
overlay and the future news meta-model see a consistent shape regardless of
which engine produced the score. ``event_type`` is drawn from the event
playbook vocabulary (config/event_playbook.yaml).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.utils.config_loader import get_event_playbook

# All valid event types come from the playbook so the two never drift apart.
EVENT_TYPES: list[str] = list(get_event_playbook().get("events", {}).keys())
DIRECTIONS = ("bearish", "bullish", "neutral")

# Scope of the event (used by the overlay to route company vs macro alphas).
SCOPES = ("company", "sector", "macro", "unknown")


def _clip(x: Any, lo: float, hi: float, default: float) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


@dataclass
class NewsScore:
    """One structured, scored news item.

    Score conventions:
        sentiment_score          : -1 (very bearish) .. +1 (very bullish)
        relevance_score          :  0 .. 1  (relevance to the target ticker)
        credibility_score        :  0 .. 1  (source quality / reliability)
        recency_score            :  0 .. 1  (1 = just now, decays with age)
        company_specificity_score:  0 .. 1  (1 = idiosyncratic, 0 = broad-market)
        expected_impact_score    :  0 .. 1  (magnitude of expected price impact)
        confidence_score         :  0 .. 1  (scorer's confidence in this row)
        expected_direction       : bearish | bullish | neutral
    """

    event_type: str = "unknown"
    scope: str = "unknown"
    sentiment_score: float = 0.0
    relevance_score: float = 0.0
    credibility_score: float = 0.5
    recency_score: float = 0.5
    company_specificity_score: float = 0.0
    expected_direction: str = "neutral"
    expected_impact_score: float = 0.0
    confidence_score: float = 0.0
    raw_summary: str = ""
    source_url: str = ""
    published_at: str | None = None          # ISO ts of the article, if known
    scored_at: str | None = None             # ISO ts when scored
    scorer: str = "heuristic"                # provenance: heuristic | gemini
    extra: dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "NewsScore":
        """Coerce all fields into valid ranges / vocabularies."""
        et = self.event_type if self.event_type in EVENT_TYPES else "unknown"
        scope = self.scope if self.scope in SCOPES else _scope_for(et)
        direction = self.expected_direction if self.expected_direction in DIRECTIONS else "neutral"
        return NewsScore(
            event_type=et,
            scope=scope,
            sentiment_score=_clip(self.sentiment_score, -1.0, 1.0, 0.0),
            relevance_score=_clip(self.relevance_score, 0.0, 1.0, 0.0),
            credibility_score=_clip(self.credibility_score, 0.0, 1.0, 0.5),
            recency_score=_clip(self.recency_score, 0.0, 1.0, 0.5),
            company_specificity_score=_clip(self.company_specificity_score, 0.0, 1.0, 0.0),
            expected_direction=direction,
            expected_impact_score=_clip(self.expected_impact_score, 0.0, 1.0, 0.0),
            confidence_score=_clip(self.confidence_score, 0.0, 1.0, 0.0),
            raw_summary=str(self.raw_summary)[:1000],
            source_url=str(self.source_url),
            published_at=self.published_at,
            scored_at=self.scored_at,
            scorer=self.scorer,
            extra=self.extra or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self.normalized())


def _scope_for(event_type: str) -> str:
    """Look up an event's scope from the playbook (defaults to 'unknown')."""
    pb = get_event_playbook().get("events", {})
    return pb.get(event_type, {}).get("scope", "unknown")


# JSON schema description handed to the LLM so it returns exactly these fields.
LLM_OUTPUT_INSTRUCTIONS = f"""
Return a JSON object with EXACTLY these keys:
- "event_type": one of {EVENT_TYPES}
- "scope": one of {list(SCOPES)}
- "sentiment_score": float in [-1, 1] (negative = bearish for the stock)
- "relevance_score": float in [0, 1]
- "credibility_score": float in [0, 1]
- "company_specificity_score": float in [0, 1] (1 = stock-specific, 0 = broad market)
- "expected_direction": one of {list(DIRECTIONS)}
- "expected_impact_score": float in [0, 1]
- "confidence_score": float in [0, 1]
- "raw_summary": one-sentence summary of the news item
Do not include any text outside the JSON object.
""".strip()
