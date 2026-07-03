"""Phase 2 tests: news schema + heuristic scorer (offline, deterministic)."""
from __future__ import annotations

from src.news.heuristic_scorer import score_item
from src.news.schema import EVENT_TYPES, NewsScore


def test_schema_clips_and_normalizes():
    s = NewsScore(
        event_type="not_a_real_event",
        sentiment_score=5.0,          # out of range -> clip to 1
        relevance_score=-3.0,         # -> clip to 0
        expected_direction="sideways",  # invalid -> neutral
        confidence_score=0.5,
    ).normalized()
    assert s.event_type == "unknown"
    assert s.sentiment_score == 1.0
    assert s.relevance_score == 0.0
    assert s.expected_direction == "neutral"


def test_event_types_nonempty():
    assert len(EVENT_TYPES) > 10
    assert "guidance_cut" in EVENT_TYPES
    assert "unknown" in EVENT_TYPES


def test_heuristic_classifies_downgrade_bearish():
    item = {
        "title": "Analyst downgrade hits NVIDIA on weak demand",
        "content": "Shares fall after a downgrade citing demand weakness.",
        "url": "https://www.reuters.com/markets/x",
        "published_date": "2026-05-29T14:00:00Z",
    }
    s = score_item(item, "NVDA", "NVIDIA Corporation").to_dict()
    assert s["event_type"] == "analyst_downgrade"
    assert s["scope"] == "company"
    assert s["expected_direction"] == "bearish"
    assert s["credibility_score"] >= 0.9          # reuters
    assert 0.0 <= s["expected_impact_score"] <= 1.0


def test_heuristic_unknown_event_defaults():
    item = {"title": "NVIDIA hosts annual charity gala", "content": "A social event.",
            "url": "https://example-blog.com/x"}
    s = score_item(item, "NVDA").to_dict()
    assert s["event_type"] in EVENT_TYPES
    assert s["credibility_score"] < 0.6           # unknown domain -> low


def test_heuristic_recency_decay():
    recent = {"title": "x", "content": "y", "url": "z",
              "published_date": "2026-05-30T00:00:00Z"}
    old = {"title": "x", "content": "y", "url": "z",
           "published_date": "2026-05-01T00:00:00Z"}
    from src.utils.timestamp_utils import to_utc
    asof = to_utc("2026-05-30T01:00:00Z")
    s_recent = score_item(recent, "NVDA", as_of=asof).to_dict()
    s_old = score_item(old, "NVDA", as_of=asof).to_dict()
    assert s_recent["recency_score"] > s_old["recency_score"]
