"""Fundamentals Agent (Council 1, spec §8.4).

Collects company financial context from Alpha Vantage (company overview +
quarterly earnings) and Finnhub (basic financials + earnings calendar).

CRITICAL timestamp rule: quarterly fundamentals may only be used for
predictions made AFTER their public release. This agent records the latest
reported earnings date and the next earnings date so the feature layer
(Phase 3) can enforce that rule. It does not itself build features.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from src.utils.api_clients import alpha_vantage_get, get_finnhub_client
from src.utils.data_loader import save_json
from src.utils.logging_utils import get_logger
from src.utils.timestamp_utils import now_utc, to_utc

log = get_logger("agents.fundamentals")


def _av_overview(ticker: str) -> dict[str, Any]:
    data = alpha_vantage_get({"function": "OVERVIEW", "symbol": ticker})
    if not data or "Symbol" not in data:
        return {}
    keys = ["PERatio", "PriceToBookRatio", "ProfitMargin", "OperatingMarginTTM",
            "ReturnOnEquityTTM", "QuarterlyRevenueGrowthYOY", "QuarterlyEarningsGrowthYOY",
            "EPS", "Beta", "MarketCapitalization", "LatestQuarter"]
    return {k: data.get(k) for k in keys if k in data}


def _av_earnings(ticker: str) -> dict[str, Any]:
    """Latest reported quarterly earnings (date + EPS surprise) from Alpha Vantage."""
    data = alpha_vantage_get({"function": "EARNINGS", "symbol": ticker})
    q = (data or {}).get("quarterlyEarnings") or []
    if not q:
        return {}
    latest = q[0]  # AV returns most-recent first
    return {
        "last_reported_date": latest.get("reportedDate"),
        "last_fiscal_ending": latest.get("fiscalDateEnding"),
        "reported_eps": latest.get("reportedEPS"),
        "estimated_eps": latest.get("estimatedEPS"),
        "eps_surprise": latest.get("surprise"),
        "eps_surprise_pct": latest.get("surprisePercentage"),
    }


def _finnhub_basics(ticker: str) -> dict[str, Any]:
    client = get_finnhub_client()
    if client is None:
        return {}
    try:
        data = client.company_basic_financials(ticker, "all")
    except Exception as exc:
        log.warning("Finnhub basic_financials failed for %s: %s", ticker, exc)
        return {}
    m = (data or {}).get("metric", {}) or {}
    keep = ["currentRatioQuarterly", "totalDebt/totalEquityQuarterly", "grossMarginTTM",
            "netProfitMarginTTM", "operatingMarginTTM", "revenueGrowthTTMYoy",
            "freeCashFlowTTM", "peTTM", "52WeekPriceReturnDaily"]
    return {k: m.get(k) for k in keep if k in m}


def _next_earnings_date(ticker: str) -> str | None:
    client = get_finnhub_client()
    if client is None:
        return None
    today = now_utc()
    frm = today.strftime("%Y-%m-%d")
    to = (today + timedelta(days=90)).strftime("%Y-%m-%d")
    try:
        cal = client.earnings_calendar(_from=frm, to=to, symbol=ticker, international=False)
    except Exception as exc:
        log.warning("Finnhub earnings_calendar failed for %s: %s", ticker, exc)
        return None
    rows = (cal or {}).get("earningsCalendar", []) or []
    dates = sorted(r.get("date") for r in rows if r.get("date"))
    return dates[0] if dates else None


def _earnings_distance_days(next_date: str | None, as_of: datetime) -> int | None:
    if not next_date:
        return None
    try:
        nd = to_utc(datetime.fromisoformat(next_date))
    except ValueError:
        return None
    return (nd - to_utc(as_of)).days


def collect_fundamentals(
    ticker: str,
    as_of: datetime | None = None,
    save: bool = True,
) -> dict[str, Any]:
    """Collect fundamentals + earnings dates. Returns a structured dict.

    Missing providers yield empty sub-dicts; never fatal.
    """
    ticker = ticker.upper()
    as_of = as_of or now_utc()

    overview = _av_overview(ticker)
    earnings = _av_earnings(ticker)
    basics = _finnhub_basics(ticker)
    next_earnings = _next_earnings_date(ticker)

    payload = {
        "ticker": ticker,
        "as_of": as_of.isoformat(),
        "overview": overview,
        "earnings": earnings,
        "basic_financials": basics,
        "next_earnings_date": next_earnings,
        "earnings_date_distance_days": _earnings_distance_days(next_earnings, as_of),
        # last_reported_date lets the feature layer enforce the post-release rule.
        "last_reported_earnings_date": earnings.get("last_reported_date"),
    }
    if save:
        save_json(payload, "fundamentals", f"{ticker}_fundamentals")
    log.info("Fundamentals for %s: overview=%d basics=%d next_earnings=%s.",
             ticker, len(overview), len(basics), next_earnings)
    return payload
