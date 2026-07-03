"""News scoring service (spec §1, §5, §6).

Turns raw news items into structured NewsScores and decision-time overlay
aggregates. Uses the Gemini provider (cost-capped, circuit-broken) for the
top-N most relevant/recent items and the always-available heuristic scorer for
the rest / on any LLM failure. Persists scores to ``structured_news_scores``.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from typing import Any

from src.core.config import get_data_sources
from src.core.logging import get_logger
from src.core.time import now_utc
from src.db import repositories as repo
from src.db.session import get_session
from src.features.news_features import aggregate_company_news, aggregate_macro_news
from src.news.heuristic_scorer import _recency, score_item as heuristic_score
from src.news.schema import LLM_OUTPUT_INSTRUCTIONS, NewsScore, _scope_for
from src.providers import gemini_provider

log = get_logger("services.news_scoring")


class NewsScoringService:
    def _prompt(self, item: dict, ticker: str, company: str | None) -> str:
        name = company or ticker
        content = (item.get("content") or "")[:1500]
        return (f"You are a sell-side risk analyst scoring a news item for its 12-24h "
                f"DOWNSIDE risk impact on {name} (ticker {ticker}).\n\n"
                f"HEADLINE: {item.get('title','')}\nSOURCE URL: {item.get('url','')}\n"
                f"EXCERPT: {content}\n\n{LLM_OUTPUT_INSTRUCTIONS}")

    def _gemini_score(self, item, ticker, company, as_of) -> NewsScore | None:
        raw = gemini_provider.generate_json(self._prompt(item, ticker, company))
        if not raw:
            return None
        data = self._parse(raw)
        if not data:
            return None
        et = data.get("event_type", "unknown")
        return NewsScore(
            event_type=et, scope=data.get("scope") or _scope_for(et),
            sentiment_score=data.get("sentiment_score", 0.0),
            relevance_score=data.get("relevance_score", 0.0),
            credibility_score=data.get("credibility_score", 0.5),
            company_specificity_score=data.get("company_specificity_score", 0.0),
            expected_direction=data.get("expected_direction", "neutral"),
            expected_impact_score=data.get("expected_impact_score", 0.0),
            confidence_score=data.get("confidence_score", 0.0),
            raw_summary=data.get("raw_summary", item.get("title", "")),
            source_url=item.get("url", ""),
            published_at=str(item.get("published_date") or "") or None,
            scored_at=as_of.isoformat(), scorer="gemini",
        ).normalized()

    @staticmethod
    def _parse(raw: str) -> dict | None:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw[4:].strip() if raw.lower().startswith("json") else raw
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            a, b = raw.find("{"), raw.rfind("}")
            if 0 <= a < b:
                try:
                    return json.loads(raw[a:b + 1])
                except json.JSONDecodeError:
                    return None
        return None

    def _priority(self, item: dict, as_of: datetime) -> float:
        rel = item.get("score")
        rel = float(rel) if isinstance(rel, (int, float)) else 0.3
        rec, _ = _recency(item.get("published_date"), as_of)
        return 0.6 * rel + 0.4 * rec

    def score(self, items: list[dict], ticker: str, company: str | None = None,
              as_of: datetime | None = None, scope: str = "company") -> list[NewsScore]:
        as_of = as_of or now_utc()
        if not items:
            return []
        max_llm = int(get_data_sources().get("news_scoring", {}).get("max_llm_items", 20))
        order = sorted(range(len(items)), key=lambda i: self._priority(items[i], as_of), reverse=True)
        llm_ok = set(order[:max_llm]) if gemini_provider.available() else set()

        out: list[NewsScore] = []
        for i, it in enumerate(items):
            s = self._gemini_score(it, ticker, company, as_of) if i in llm_ok else None
            if s is None:
                s = heuristic_score(it, ticker, company, as_of)
            # deterministic recency regardless of scorer
            rec, pub = _recency(it.get("published_date"), as_of)
            s.recency_score = rec
            if pub and not s.published_at:
                s.published_at = pub
            out.append(s.normalized())
        log.info("news_scored", ticker=ticker, scope=scope, n=len(out),
                 scorers=dict(Counter(s.scorer for s in out)))
        return out

    def persist(self, scores: list[NewsScore], ticker: str) -> int:
        if not scores:
            return 0
        rows = []
        for s in scores:
            d = s.to_dict()
            rows.append({
                "ticker": ticker, "scope": d["scope"], "event_type": d["event_type"],
                "sentiment_score": d["sentiment_score"], "relevance_score": d["relevance_score"],
                "credibility_score": d["credibility_score"], "recency_score": d["recency_score"],
                "company_specificity_score": d["company_specificity_score"],
                "expected_direction": d["expected_direction"],
                "expected_impact_score": d["expected_impact_score"],
                "confidence_score": d["confidence_score"], "scorer": d["scorer"],
                "source_url": d.get("source_url"),
                "published_at": _dt(d.get("published_at")), "scored_at": _dt(d.get("scored_at")) or now_utc(),
            })
        with get_session() as sess:
            repo.save_news_scores(sess, rows)
        return len(rows)

    def overlay_features(self, company_scores: list[NewsScore],
                         macro_scores: list[NewsScore]) -> dict[str, Any]:
        comp = aggregate_company_news([s.to_dict() for s in company_scores])
        macro = aggregate_macro_news([s.to_dict() for s in macro_scores])
        return {**comp, **macro}


def _dt(v):
    if not v:
        return None
    import pandas as pd
    try:
        return pd.to_datetime(v, utc=True).to_pydatetime()
    except Exception:
        return None
