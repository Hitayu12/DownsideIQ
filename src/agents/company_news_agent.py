"""Company-Specific News Agent (Council 1, spec §8.2).

Gathers company-level catalysts (earnings, guidance, downgrades, lawsuits,
product news, M&A) from Tavily and (when available) Finnhub company news,
scores each into the structured schema, and computes a news-volume signal.
Output feeds the company side of the live news overlay.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from src.news.schema import NewsScore
from src.news.scorer import score_items
from src.news.search import dedupe, search_news
from src.utils.api_clients import get_finnhub_client
from src.utils.config_loader import get_ticker_config
from src.utils.data_loader import save_json
from src.utils.logging_utils import get_logger
from src.utils.timestamp_utils import now_utc, to_utc

log = get_logger("agents.company_news")

# Baseline daily article count; above this we flag abnormal media attention.
_ABNORMAL_VOLUME_THRESHOLD = 10


def _company_queries(ticker: str, company: str | None) -> list[str]:
    name = company or ticker
    return [
        f"{name} stock news",
        f"{name} earnings guidance",
        f"{ticker} analyst rating downgrade upgrade",
        f"{name} lawsuit regulation investigation",
        f"{name} product launch delay",
    ]


def _finnhub_company_news(ticker: str, days: int) -> list[dict[str, Any]]:
    client = get_finnhub_client()
    if client is None:
        return []
    today = now_utc()
    frm = (today - timedelta(days=days)).strftime("%Y-%m-%d")
    to = today.strftime("%Y-%m-%d")
    try:
        rows = client.company_news(ticker, _from=frm, to=to)
    except Exception as exc:
        log.warning("Finnhub company_news failed for %s: %s", ticker, exc)
        return []
    items: list[dict[str, Any]] = []
    for r in rows or []:
        ts = r.get("datetime")
        pub = to_utc(datetime.fromtimestamp(ts)).isoformat() if ts else None
        items.append({
            "title": r.get("headline", ""),
            "url": r.get("url", ""),
            "content": r.get("summary", ""),
            "score": None,
            "published_date": pub,
            "source": r.get("source"),
            "query": "finnhub_company_news",
        })
    return items


def collect_company_news(
    ticker: str,
    max_results_per_query: int = 4,
    days: int = 3,
    as_of: datetime | None = None,
    save: bool = True,
) -> dict[str, Any]:
    """Collect + score company news. Returns {'raw','scored','news_volume','abnormal_news_volume'}."""
    ticker = ticker.upper()
    as_of = as_of or now_utc()
    try:
        company = get_ticker_config(ticker).get("name")
    except KeyError:
        company = None

    raw: list[dict[str, Any]] = []
    for q in _company_queries(ticker, company):
        raw.extend(search_news(q, max_results=max_results_per_query, days=days))
    raw.extend(_finnhub_company_news(ticker, days=days))
    raw = dedupe(raw)

    scored: list[NewsScore] = score_items(raw, ticker, company=company, as_of=as_of)
    scored_dicts = [s.to_dict() for s in scored]

    news_volume = len(raw)
    abnormal = news_volume >= _ABNORMAL_VOLUME_THRESHOLD

    payload = {
        "ticker": ticker,
        "company": company,
        "as_of": as_of.isoformat(),
        "news_volume": news_volume,
        "abnormal_news_volume": abnormal,
        "raw": raw,
        "scored": scored_dicts,
    }
    if save:
        save_json(payload, "news", f"{ticker}_company_news")
    log.info("Company news for %s: %d raw -> %d scored (volume=%d, abnormal=%s).",
             ticker, len(raw), len(scored_dicts), news_volume, abnormal)
    return payload
