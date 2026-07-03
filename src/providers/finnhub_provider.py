"""Finnhub provider — company news, basic financials, earnings calendar.

Failure rule (spec §6): degrade gracefully (return [] / {} / None), never raise.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any

from src.core.config import env
from src.core.logging import get_logger
from src.core.time import now_utc, to_utc
from src.providers.base import call_with_retry, provider_cfg

log = get_logger("providers.finnhub")


@lru_cache(maxsize=1)
def _client():
    e = env()
    if not e.finnhub_api_key:
        log.warning("finnhub_key_missing")
        return None
    try:
        import finnhub

        return finnhub.Client(api_key=e.finnhub_api_key)
    except Exception as exc:  # pragma: no cover
        log.warning("finnhub_init_failed", error=str(exc)[:160])
        return None


def available() -> bool:
    return _client() is not None


def _retry(fn, label):
    cfg = provider_cfg("fundamentals")
    return call_with_retry(fn, provider=f"finnhub.{label}",
                           retries=int(cfg.get("retries", 2)),
                           backoff=float(cfg.get("retry_backoff_seconds", 3)))


def company_news(ticker: str, days: int = 3) -> list[dict[str, Any]]:
    client = _client()
    if client is None:
        return []
    today = now_utc()
    frm = (today - timedelta(days=days)).strftime("%Y-%m-%d")
    to = today.strftime("%Y-%m-%d")
    try:
        rows = _retry(lambda: client.company_news(ticker, _from=frm, to=to), "company_news")
    except Exception as exc:
        log.warning("finnhub_company_news_failed", ticker=ticker, error=str(exc)[:160])
        return []
    out = []
    for r in rows or []:
        ts = r.get("datetime")
        pub = to_utc(datetime.fromtimestamp(ts)).isoformat() if ts else None
        out.append({"title": r.get("headline", ""), "url": r.get("url", ""),
                    "content": r.get("summary", ""), "score": None,
                    "published_date": pub, "source": r.get("source"),
                    "query": "finnhub_company_news"})
    return out


def basic_financials(ticker: str) -> dict[str, Any]:
    client = _client()
    if client is None:
        return {}
    try:
        data = _retry(lambda: client.company_basic_financials(ticker, "all"), "basic_financials")
    except Exception as exc:
        log.warning("finnhub_basics_failed", ticker=ticker, error=str(exc)[:160])
        return {}
    return (data or {}).get("metric", {}) or {}


def next_earnings_date(ticker: str) -> str | None:
    client = _client()
    if client is None:
        return None
    today = now_utc()
    frm = today.strftime("%Y-%m-%d")
    to = (today + timedelta(days=90)).strftime("%Y-%m-%d")
    try:
        cal = _retry(lambda: client.earnings_calendar(_from=frm, to=to, symbol=ticker,
                                                      international=False), "earnings_calendar")
    except Exception as exc:
        log.warning("finnhub_earnings_failed", ticker=ticker, error=str(exc)[:160])
        return None
    rows = (cal or {}).get("earningsCalendar", []) or []
    dates = sorted(r.get("date") for r in rows if r.get("date"))
    return dates[0] if dates else None
