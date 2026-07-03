"""News-feature aggregation (spec §8.1/§8.2 → §9; hybrid overlay).

Collapses the list of structured NewsScore items (company + macro) into a small
set of aggregate scores that feed the live news overlay. Each item is weighted
by relevance × credibility × recency, so stale / low-quality / off-topic items
contribute little. These are LIVE features only — they are never backfilled into
the historical training table (see architecture decision).

Sign convention (matches the overlay): higher *_risk_score = more DOWNSIDE risk.
"""
from __future__ import annotations

from typing import Any

import numpy as np

_DIR_SIGN = {"bearish": 1.0, "bullish": -1.0, "neutral": 0.0}


def _weight(item: dict[str, Any]) -> float:
    return (
        float(item.get("relevance_score", 0.0))
        * float(item.get("credibility_score", 0.0))
        * float(item.get("recency_score", 0.0))
    )


def _item_downside(item: dict[str, Any]) -> float:
    """Per-item downside contribution in ~[-1, 1] (positive = bearish/downside)."""
    sentiment_dn = -float(item.get("sentiment_score", 0.0))           # neg sentiment = downside
    dir_dn = _DIR_SIGN.get(item.get("expected_direction", "neutral"), 0.0)
    impact = float(item.get("expected_impact_score", 0.0))
    return 0.5 * sentiment_dn + 0.5 * dir_dn * impact


def aggregate_company_news(scored: list[dict[str, Any]], volume_threshold: int = 10) -> dict[str, float]:
    """Aggregate scored company-news items into overlay features."""
    if not scored:
        return {
            "company_news_risk_score": 0.0,
            "negative_catalyst_score": 0.0,
            "positive_catalyst_score": 0.0,
            "company_specificity_score": 0.0,
            "news_volume_score": 0.0,
            "company_news_confidence": 0.0,
            "company_news_count": 0,
        }
    weights = np.array([_weight(it) for it in scored])
    downside = np.array([_item_downside(it) for it in scored])
    w_sum = weights.sum() or 1.0

    net_risk = float((weights * downside).sum() / w_sum)             # [-1, 1]
    neg = float((weights * np.clip(downside, 0, None)).sum() / w_sum)
    pos = float((weights * np.clip(-downside, 0, None)).sum() / w_sum)
    specificity = float(
        (weights * np.array([it.get("company_specificity_score", 0.0) for it in scored])).sum() / w_sum
    )
    confidence = float(np.mean([it.get("confidence_score", 0.0) for it in scored]))

    return {
        "company_news_risk_score": float(np.clip(net_risk, -1, 1)),
        "negative_catalyst_score": float(np.clip(neg, 0, 1)),
        "positive_catalyst_score": float(np.clip(pos, 0, 1)),
        "company_specificity_score": float(np.clip(specificity, 0, 1)),
        "news_volume_score": float(min(1.0, len(scored) / max(1, volume_threshold))),
        "company_news_confidence": float(np.clip(confidence, 0, 1)),
        "company_news_count": len(scored),
    }


def aggregate_macro_news(scored: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate scored macro-news items into overlay features."""
    if not scored:
        return {
            "macro_risk_score": 0.0,
            "macro_sentiment_score": 0.0,
            "macro_confidence_score": 0.0,
            "macro_news_count": 0,
        }
    weights = np.array([_weight(it) for it in scored])
    downside = np.array([_item_downside(it) for it in scored])
    w_sum = weights.sum() or 1.0

    net_risk = float((weights * downside).sum() / w_sum)
    sentiment = float(
        (weights * np.array([it.get("sentiment_score", 0.0) for it in scored])).sum() / w_sum
    )
    confidence = float(np.mean([it.get("confidence_score", 0.0) for it in scored]))
    return {
        "macro_risk_score": float(np.clip(net_risk, -1, 1)),
        "macro_sentiment_score": float(np.clip(sentiment, -1, 1)),
        "macro_confidence_score": float(np.clip(confidence, 0, 1)),
        "macro_news_count": len(scored),
    }
