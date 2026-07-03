"""Ingestion service — pulls + validates + persists raw inputs (spec §1, §5).

Uses the hardened providers. Enforces the hard data rule: a primary-price
failure raises ``DataQualityError`` (signal blocked). News / fundamentals failures
degrade gracefully and are recorded on a ``DegradedMode`` surfaced downstream.
Raw price/news rows are persisted for auditability.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.core.config import get_ticker_config
from src.core.logging import get_logger
from src.core.time import now_utc
from src.db import repositories as repo
from src.db.session import get_session
from src.domain.features import IngestionResult
from src.providers import alpha_vantage, finnhub_provider, prices, tavily_provider

log = get_logger("services.ingestion")

_MACRO_QUERIES = [
    "Federal Reserve interest rate decision", "US CPI inflation report",
    "US jobs report nonfarm payrolls", "Treasury yields move stock market",
    "US recession risk economy", "geopolitical risk markets tariffs",
    "semiconductor sector selloff", "stock market risk off selloff",
]


def _company_queries(ticker: str, name: str | None) -> list[str]:
    n = name or ticker
    return [f"{n} stock news", f"{n} earnings guidance",
            f"{ticker} analyst rating downgrade upgrade",
            f"{n} lawsuit regulation investigation", f"{n} product launch delay"]


class IngestionService:
    def ingest(self, ticker: str, bar_size: str | None = None, *, as_of: datetime | None = None,
               persist: bool = True) -> IngestionResult:
        ticker = ticker.upper()
        as_of = as_of or now_utc()
        cfg = get_ticker_config(ticker)
        bar_size = bar_size or "1d"
        result = IngestionResult(ticker=ticker, as_of=as_of, bar_size=bar_size, prices=pd.DataFrame())

        # --- PRIMARY price (hard: failure raises DataQualityError) ---
        result.prices = prices.fetch_ohlcv(ticker, bar_size=bar_size, require=True)

        # --- context assets (soft) ---
        assets = [*cfg.get("market_etfs", []), *cfg.get("sector_etfs", []),
                  *cfg.get("peers", []), cfg.get("vol_proxy", "^VIX")]
        for sym in dict.fromkeys(a for a in assets if a):
            df = prices.fetch_ohlcv(sym, bar_size=bar_size, require=False)
            if not df.empty:
                result.context[sym.replace("^", "_")] = df

        # --- news (soft -> price-only mode if unavailable) ---
        if tavily_provider.available():
            for q in _MACRO_QUERIES:
                result.raw_macro_news += tavily_provider.search_news(q)
            for q in _company_queries(ticker, cfg.get("name")):
                result.raw_company_news += tavily_provider.search_news(q)
        else:
            result.degraded.degrade("news", "tavily unavailable")
        result.raw_company_news += finnhub_provider.company_news(ticker)
        if not (result.raw_macro_news or result.raw_company_news):
            result.degraded.degrade("news", "no news returned")

        # --- fundamentals (soft) ---
        result.fundamentals = self._fundamentals(ticker, as_of)
        if not result.fundamentals.get("overview") and not result.fundamentals.get("basic_financials"):
            result.degraded.degrade("fundamentals", "no fundamentals available")

        if persist:
            self._persist(result)
        log.info("ingested", ticker=ticker, bars=len(result.prices),
                 context=len(result.context), macro=len(result.raw_macro_news),
                 company=len(result.raw_company_news), data_quality=result.degraded.status.value)
        return result

    def _fundamentals(self, ticker: str, as_of: datetime) -> dict:
        overview, earnings = {}, {}
        ov = alpha_vantage.query({"function": "OVERVIEW", "symbol": ticker})
        if ov and "Symbol" in ov:
            keys = ["PERatio", "ProfitMargin", "OperatingMarginTTM", "QuarterlyRevenueGrowthYOY",
                    "Beta", "EPS", "MarketCapitalization", "LatestQuarter"]
            overview = {k: ov.get(k) for k in keys if k in ov}
        ea = alpha_vantage.query({"function": "EARNINGS", "symbol": ticker})
        q = (ea or {}).get("quarterlyEarnings") or []
        if q:
            earnings = {"last_reported_date": q[0].get("reportedDate"),
                        "eps_surprise": q[0].get("surprise"),
                        "eps_surprise_pct": q[0].get("surprisePercentage")}
        next_earnings = finnhub_provider.next_earnings_date(ticker)
        dist = None
        if next_earnings:
            from src.core.time import to_utc
            try:
                dist = (to_utc(datetime.fromisoformat(next_earnings)) - to_utc(as_of)).days
            except ValueError:
                dist = None
        return {
            "overview": overview, "earnings": earnings,
            "basic_financials": finnhub_provider.basic_financials(ticker),
            "next_earnings_date": next_earnings, "earnings_date_distance_days": dist,
            "last_reported_earnings_date": earnings.get("last_reported_date"),
        }

    def _persist(self, r: IngestionResult) -> None:
        with get_session() as s:
            repo.upsert_price_bars(s, r.ticker, r.bar_size, prices.to_bar_rows(r.prices))
            news_rows = []
            for scope, items in (("macro", r.raw_macro_news), ("company", r.raw_company_news)):
                for it in items:
                    pub = it.get("published_date")
                    news_rows.append({
                        "ticker": r.ticker, "scope": scope, "query": it.get("query"),
                        "title": it.get("title"), "url": it.get("url"),
                        "content": it.get("content"),
                        "published_at": pd.to_datetime(pub, utc=True).to_pydatetime() if pub else None,
                        "provider": it.get("source") or "tavily", "fetched_at": r.as_of,
                        "payload": {k: it.get(k) for k in ("score", "query")},
                    })
            if news_rows:
                repo.save_raw_news(s, news_rows)
