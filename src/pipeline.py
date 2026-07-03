"""DownsideIQ pipeline orchestration (spec §23).

Phase 2 implements ``collect_data``. Later phases add ``build_features``,
``run_prediction_models``, ``generate_final_decision``, ``apply_risk_controls``,
``log_prediction``, etc. Each stage is intentionally small and composable.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from src.agents.company_news_agent import collect_company_news
from src.agents.fundamentals_agent import collect_fundamentals
from src.agents.macro_news_agent import collect_macro_news
from src.agents.market_context_agent import collect_market_context, context_assets_for
from src.agents.price_trend_agent import collect_prices
from src.features.feature_builder import build_feature_table, build_live_feature_row
from src.utils.config_loader import data_dir, get_settings
from src.utils.data_loader import load_frame, load_json, save_frame, save_json
from src.utils.logging_utils import get_logger
from src.utils.timestamp_utils import now_utc

log = get_logger("pipeline")


def collect_data(
    ticker: str,
    bar_size: str | None = None,
    as_of: datetime | None = None,
    save: bool = True,
) -> dict[str, Any]:
    """Run the full Data Intelligence Council (Council 1) for ``ticker``.

    Returns a manifest dict summarising what was collected. Every sub-collector
    degrades gracefully, so this never raises on a missing provider — it reports
    what succeeded. A manifest is saved to data/raw/<ticker>_collection_manifest.json.
    """
    ticker = ticker.upper()
    as_of = as_of or now_utc()
    settings = get_settings()
    bar_size = bar_size or settings.get("bar_size", "1d")
    log.info("=== Council 1 data collection: %s (bar=%s, as_of=%s) ===", ticker, bar_size, as_of)

    prices = collect_prices(ticker, bar_size=bar_size, save=save)
    context = collect_market_context(ticker, bar_size=bar_size, save=save)
    macro = collect_macro_news(ticker, as_of=as_of, save=save)
    company = collect_company_news(ticker, as_of=as_of, save=save)
    fundamentals = collect_fundamentals(ticker, as_of=as_of, save=save)

    manifest = {
        "ticker": ticker,
        "as_of": as_of.isoformat(),
        "bar_size": bar_size,
        "price_bars": int(len(prices)),
        "price_last_ts": str(prices.index.max()) if len(prices) else None,
        "context_assets": sorted(context.keys()),
        "context_asset_count": len(context),
        "macro_events": len(macro.get("scored", [])),
        "company_events": len(company.get("scored", [])),
        "company_news_volume": company.get("news_volume", 0),
        "abnormal_news_volume": company.get("abnormal_news_volume", False),
        "next_earnings_date": fundamentals.get("next_earnings_date"),
        "earnings_date_distance_days": fundamentals.get("earnings_date_distance_days"),
        "fundamentals_available": bool(fundamentals.get("overview") or fundamentals.get("basic_financials")),
    }
    if save:
        save_json(manifest, "", f"{ticker}_collection_manifest")
    log.info("=== Collection manifest: %s ===", manifest)
    return {
        "manifest": manifest,
        "prices": prices,
        "context": context,
        "macro": macro,
        "company": company,
        "fundamentals": fundamentals,
    }


def load_collected(ticker: str) -> dict[str, Any] | None:
    """Reconstruct a ``collected`` dict from saved raw files (no network).

    Returns None if no price data has been saved for the ticker yet.
    """
    ticker = ticker.upper()
    prices = load_frame("prices", ticker)
    if prices is None or prices.empty:
        return None
    context: dict[str, pd.DataFrame] = {}
    for sym in context_assets_for(ticker):
        safe = sym.replace("^", "_")
        df = load_frame("prices", safe)
        if df is not None and not df.empty:
            context[safe] = df
    return {
        "prices": prices,
        "context": context,
        "macro": load_json("macro", f"{ticker}_macro_news") or {},
        "company": load_json("news", f"{ticker}_company_news") or {},
        "fundamentals": load_json("fundamentals", f"{ticker}_fundamentals") or {},
    }


def build_features(
    ticker: str,
    collected: dict[str, Any] | None = None,
    as_of: datetime | None = None,
    save: bool = True,
) -> dict[str, Any]:
    """Build the historical feature table + the live feature row for ``ticker``.

    Reuses saved raw data when available; collects fresh data only if none exists.
    """
    ticker = ticker.upper()
    as_of = as_of or now_utc()
    if collected is None:
        collected = load_collected(ticker)
        if collected is None:
            log.info("No saved raw data for %s; collecting fresh.", ticker)
            collected = collect_data(ticker, as_of=as_of, save=save)

    table = build_feature_table(collected["prices"], collected.get("context", {}), ticker)
    row = build_live_feature_row(collected, ticker, as_of=as_of)

    if save and not table.empty:
        import json
        path = data_dir() / "processed" / "features"
        path.mkdir(parents=True, exist_ok=True)
        table.to_parquet(path / f"{ticker}_features.parquet")
        with (path / f"{ticker}_live_feature_row.json").open("w", encoding="utf-8") as fh:
            json.dump(row, fh, indent=2, default=str)
        log.info("Saved feature table + live row -> %s", path)
    return {"table": table, "row": row}


def generate_decision(
    ticker: str,
    collected: dict[str, Any] | None = None,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    """Full Council 2 + Final Decision Engine → SHORT / WATCH / NO TRADE.

    Reuses saved raw data when available; collects fresh only if none exists.
    """
    from src.models.ensemble_engine import decide
    from src.models.training import run_prediction_models

    ticker = ticker.upper()
    as_of = as_of or now_utc()
    if collected is None:
        collected = load_collected(ticker) or collect_data(ticker, as_of=as_of)

    model_outputs = run_prediction_models(collected, ticker, as_of=as_of)
    decision = decide(model_outputs)
    log.info("Decision for %s: %s (adj_p=%.3f, risk=%.3f).", ticker,
             decision["decision"], decision["adjusted_p_downside"],
             decision["adjusted_downside_risk_score"])
    return decision
