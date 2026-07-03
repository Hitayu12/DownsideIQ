"""Heuristic / lexicon news scorer — the always-available fallback.

No API key, no network. Classifies a news item into an event_type via keyword
matching, estimates sentiment from a finance lexicon, credibility from a source
allowlist, and recency from the publish timestamp. Coarser than the LLM scorer
but deterministic and dependency-free, so the pipeline never stalls on news.
"""
from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any

import pandas as pd

from src.news.schema import NewsScore, _scope_for
from src.utils.config_loader import get_event_playbook
from src.utils.timestamp_utils import now_utc, to_utc

# --- Event keyword map: event_type -> trigger phrases (lowercased, substring) ---
_EVENT_KEYWORDS: dict[str, list[str]] = {
    "guidance_cut": ["cuts guidance", "lowers guidance", "guidance cut", "slashes outlook", "warns on", "profit warning"],
    "guidance_raise": ["raises guidance", "lifts guidance", "boosts outlook", "raises outlook"],
    "earnings_miss": ["misses estimates", "earnings miss", "misses on revenue", "disappointing earnings", "falls short"],
    "earnings_beat": ["beats estimates", "earnings beat", "tops estimates", "blowout quarter", "beats on revenue"],
    "analyst_downgrade": ["downgrade", "downgraded", "cuts price target", "lowers price target", "sell rating", "underweight"],
    "analyst_upgrade": ["upgrade", "upgraded", "raises price target", "buy rating", "overweight", "outperform"],
    "lawsuit": ["lawsuit", "sued", "class action", "litigation", "legal action"],
    "regulatory_action": ["regulator", "regulatory", "antitrust", "ftc", "doj", "investigation", "probe", "subpoena", "export ban", "export restriction", "sanction"],
    "executive_departure": ["resigns", "steps down", "ceo departs", "cfo departs", "executive departure", "ousted"],
    "product_delay": ["delay", "delayed", "postpones", "pushed back", "recall"],
    "product_launch": ["launches", "unveils", "announces new", "new product", "new chip", "ships"],
    "partnership": ["partnership", "partners with", "collaboration", "deal with", "contract win"],
    "ma_rumor": ["acquisition", "acquire", "merger", "buyout", "takeover", "in talks to buy"],
    "supply_chain_risk": ["supply chain", "shortage", "production halt", "factory", "logistics disruption"],
    "demand_weakness": ["weak demand", "slowing demand", "demand slump", "soft demand", "order cuts"],
    "hawkish_fed": ["hawkish", "rate hike", "raises rates", "higher for longer"],
    "dovish_fed": ["dovish", "rate cut", "cuts rates", "easing"],
    "rate_shock": ["surge in yields", "yields spike", "treasury selloff", "bond rout"],
    "inflation_surprise": ["inflation", "cpi", "ppi", "hotter than expected", "price pressures"],
    "recession_risk": ["recession", "economic slowdown", "contraction", "downturn"],
    "geopolitical_risk": ["war", "conflict", "tariff", "trade war", "geopolitical", "invasion", "tensions"],
    "liquidity_credit_stress": ["credit crunch", "liquidity crisis", "bank failure", "default", "credit stress"],
    "sector_selloff": ["sector selloff", "chip stocks fall", "semiconductor selloff", "sector slump", "tech rout"],
    "abnormal_news_volume": [],  # set by the agent based on article count, not keywords
}

# --- Finance sentiment lexicon (lowercased substrings) ---
_BEARISH = ["plunge", "plunges", "tumble", "slump", "crash", "selloff", "sell-off", "drop", "falls",
            "fell", "decline", "downgrade", "miss", "weak", "warning", "cuts", "lawsuit", "probe",
            "investigation", "loss", "losses", "concern", "fears", "risk", "halt", "recall",
            "bearish", "disappoint", "shortfall", "slowdown", "ban", "restriction"]
_BULLISH = ["surge", "soar", "rally", "jump", "gains", "beat", "beats", "tops", "record", "upgrade",
            "strong", "growth", "raises", "wins", "approval", "bullish", "optimism", "boost",
            "outperform", "breakthrough", "expansion"]

# --- Source credibility allowlist (domain substring -> score) ---
_CREDIBILITY: list[tuple[str, float]] = [
    ("sec.gov", 0.98), ("federalreserve.gov", 0.98), ("bls.gov", 0.97),
    ("reuters.com", 0.95), ("bloomberg.com", 0.95), ("wsj.com", 0.93),
    ("ft.com", 0.93), ("cnbc.com", 0.88), ("apnews.com", 0.92),
    ("marketwatch.com", 0.82), ("barrons.com", 0.85), ("nytimes.com", 0.85),
    ("finance.yahoo.com", 0.7), ("yahoo.com", 0.68), ("investing.com", 0.65),
    ("seekingalpha.com", 0.6), ("fool.com", 0.5), ("benzinga.com", 0.55),
    ("prnewswire.com", 0.6), ("businesswire.com", 0.62), ("globenewswire.com", 0.6),
]
_DEFAULT_CREDIBILITY = 0.45
_RECENCY_HALFLIFE_HOURS = 18.0   # news decays fast for a 12-24h horizon


def _classify_event(text: str) -> tuple[str, float]:
    """Return (event_type, match_strength 0..1) from keyword hits."""
    best_type, best_hits = "unknown", 0
    for etype, kws in _EVENT_KEYWORDS.items():
        hits = sum(1 for kw in kws if kw in text)
        if hits > best_hits:
            best_type, best_hits = etype, hits
    strength = min(1.0, 0.4 + 0.3 * best_hits) if best_hits else 0.0
    return best_type, strength


def _sentiment(text: str) -> float:
    b = sum(1 for w in _BEARISH if re.search(rf"\b{re.escape(w)}", text))
    g = sum(1 for w in _BULLISH if re.search(rf"\b{re.escape(w)}", text))
    if b == 0 and g == 0:
        return 0.0
    return max(-1.0, min(1.0, (g - b) / (g + b)))


def _credibility(url: str) -> float:
    u = (url or "").lower()
    for domain, score in _CREDIBILITY:
        if domain in u:
            return score
    return _DEFAULT_CREDIBILITY


def _recency(published_at: Any, as_of: datetime) -> tuple[float, str | None]:
    if not published_at:
        return 0.5, None
    try:
        pub = to_utc(pd.Timestamp(published_at))
    except Exception:
        return 0.5, None
    age_h = max(0.0, (to_utc(as_of) - pub).total_seconds() / 3600.0)
    score = math.pow(0.5, age_h / _RECENCY_HALFLIFE_HOURS)
    return max(0.0, min(1.0, score)), pub.isoformat()


def _relevance(item: dict[str, Any], ticker: str, company: str | None) -> float:
    # Prefer Tavily's own relevance score if present.
    tav = item.get("score")
    if isinstance(tav, (int, float)):
        return max(0.0, min(1.0, float(tav)))
    text = f"{item.get('title','')} {item.get('content','')}".lower()
    hits = text.count(ticker.lower())
    if company:
        hits += text.count(company.lower())
    return min(1.0, 0.3 + 0.2 * hits) if hits else 0.2


def score_item(
    item: dict[str, Any],
    ticker: str,
    company: str | None = None,
    as_of: datetime | None = None,
) -> NewsScore:
    """Heuristically score one Tavily news item into a NewsScore."""
    as_of = as_of or now_utc()
    title = item.get("title", "") or ""
    content = item.get("content", "") or item.get("snippet", "") or ""
    url = item.get("url", "") or ""
    text = f"{title}. {content}".lower()

    etype, strength = _classify_event(text)
    sentiment = _sentiment(text)
    credibility = _credibility(url)
    recency, pub_iso = _recency(item.get("published_date") or item.get("published_at"), as_of)
    relevance = _relevance(item, ticker, company)

    playbook = get_event_playbook().get("events", {}).get(etype, {"bias": 0.0, "impact_weight": 0.2})
    bias = float(playbook.get("bias", 0.0))
    impact_weight = float(playbook.get("impact_weight", 0.2))
    scope = _scope_for(etype)

    # Direction: combine playbook prior bias with observed sentiment.
    blended = 0.6 * (-bias) + 0.4 * sentiment   # -bias because positive bias = bearish
    if blended < -0.15:
        direction = "bearish"
    elif blended > 0.15:
        direction = "bullish"
    else:
        direction = "neutral"

    specificity = 1.0 if scope == "company" else (0.5 if scope == "sector" else 0.1)
    expected_impact = min(1.0, impact_weight * (0.5 + 0.5 * credibility) * (0.5 + 0.5 * relevance))
    confidence = min(1.0, 0.25 + 0.4 * strength + 0.2 * credibility + 0.15 * relevance)

    return NewsScore(
        event_type=etype,
        scope=scope,
        sentiment_score=sentiment,
        relevance_score=relevance,
        credibility_score=credibility,
        recency_score=recency,
        company_specificity_score=specificity,
        expected_direction=direction,
        expected_impact_score=expected_impact,
        confidence_score=confidence,
        raw_summary=title or content[:160],
        source_url=url,
        published_at=pub_iso,
        scored_at=as_of.isoformat() if isinstance(as_of, datetime) else None,
        scorer="heuristic",
    ).normalized()
