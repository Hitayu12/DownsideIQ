"""Layer 5 — news scoring service (offline; heuristic path + parsing + aggregation)."""
from __future__ import annotations

from src.services.news_scoring_service import NewsScoringService
from src.providers import gemini_provider


def _items():
    return [
        {"title": "NVIDIA analyst downgrade on weak demand", "content": "Downgrade, shares fall.",
         "url": "https://www.reuters.com/x", "published_date": "2026-05-29T14:00:00Z", "score": 0.9},
        {"title": "NVIDIA opens new office", "content": "Routine corporate update.",
         "url": "https://smallblog.com/y", "published_date": "2026-05-20T00:00:00Z", "score": 0.2},
    ]


def test_heuristic_scoring_when_llm_unavailable(monkeypatch):
    monkeypatch.setattr(gemini_provider, "available", lambda: False)
    svc = NewsScoringService()
    scores = svc.score(_items(), "NVDA", "NVIDIA Corporation", scope="company")
    assert len(scores) == 2
    assert all(s.scorer == "heuristic" for s in scores)
    assert scores[0].event_type == "analyst_downgrade"
    assert scores[0].expected_direction == "bearish"


def test_overlay_aggregation_signs(monkeypatch):
    monkeypatch.setattr(gemini_provider, "available", lambda: False)
    svc = NewsScoringService()
    company = svc.score(_items(), "NVDA", "NVIDIA Corporation", scope="company")
    agg = svc.overlay_features(company, [])
    assert "company_news_risk_score" in agg and "macro_risk_score" in agg
    assert agg["negative_catalyst_score"] >= 0.0


def test_parse_handles_fenced_json():
    raw = '```json\n{"event_type": "guidance_cut", "sentiment_score": -0.8}\n```'
    data = NewsScoringService._parse(raw)
    assert data["event_type"] == "guidance_cut"


def test_empty_items_returns_empty(monkeypatch):
    monkeypatch.setattr(gemini_provider, "available", lambda: False)
    assert NewsScoringService().score([], "NVDA") == []
