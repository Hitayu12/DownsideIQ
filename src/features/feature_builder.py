"""Feature assembly (Council 1 → §9 feature row).

Two products:

1. ``build_feature_table`` — the BACKFILLABLE historical feature matrix used to
   train the models (price + volume + volatility + market-context). It contains
   NO news and NO snapshot-fundamentals, because those cannot be reconstructed
   leakage-free for the past (architecture decision).

2. ``build_live_feature_row`` — one clean feature row for *now*, which augments
   the latest backfillable features with the live news overlay aggregates,
   timestamp-safe fundamentals, and a data-confidence score. This is what the
   decision engine consumes at prediction time.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.features.fundamental_features import compute_fundamental_features
from src.features.market_context_features import compute_market_context_features
from src.features.news_features import aggregate_company_news, aggregate_macro_news
from src.features.price_features import compute_price_features
from src.features.volatility_features import compute_volatility_features
from src.utils.config_loader import get_settings, get_ticker_config
from src.utils.logging_utils import get_logger
from src.utils.timestamp_utils import now_utc

log = get_logger("features.builder")

# Core backfillable feature columns the models can rely on.
CORE_FEATURE_COLS = [
    "log_return_1b", "return_1b", "return_5b", "return_10b", "return_20b",
    "momentum_score", "mean_reversion_score", "moving_average_spread",
    "drawdown_from_recent_high", "intraday_range", "gap_risk_score",
    "volume_z_score", "abnormal_volume_score",
    "rolling_volatility_12b", "rolling_volatility_24b", "realized_volatility",
    "downside_volatility", "volatility_ratio",
    "market_return_1b", "market_return_5b", "sector_return_1b",
    "market_beta", "sector_beta", "correlation_to_market", "correlation_to_sector",
    "relative_strength_vs_market", "relative_strength_vs_sector",
    "correlation_spike_score", "stock_specific_move_score", "vix_level",
]


def build_feature_table(
    prices: pd.DataFrame,
    context: dict[str, pd.DataFrame],
    ticker: str,
) -> pd.DataFrame:
    """Assemble the backfillable historical feature matrix (for training)."""
    if prices.empty:
        return pd.DataFrame()
    ticker_cfg = get_ticker_config(ticker)

    price_f = compute_price_features(prices)
    vol_f = compute_volatility_features(prices)
    mkt_f = compute_market_context_features(prices, context, ticker_cfg)

    table = pd.concat([price_f, vol_f, mkt_f], axis=1)
    # Carry through regime labels (object dtype) without numeric coercion.
    table = table.loc[:, ~table.columns.duplicated()]
    table.index.name = "timestamp"
    log.info("Built feature table for %s: %d rows × %d cols.",
             ticker, len(table), table.shape[1])
    return table


def compute_data_confidence_score(
    feature_row: dict[str, Any],
    company_news: dict[str, Any],
    macro_news: dict[str, Any],
    fundamentals_available: bool,
    price_age_bars: int,
) -> tuple[float, bool]:
    """Return (data_confidence_score 0..1, missing_data_flag).

    Combines feature completeness, news quality/recency, fundamentals
    availability, and price freshness (spec §11.4).
    """
    core_vals = [feature_row.get(c) for c in CORE_FEATURE_COLS]
    present = [v for v in core_vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    completeness = len(present) / len(CORE_FEATURE_COLS)

    has_company = company_news.get("company_news_count", 0) > 0
    has_macro = macro_news.get("macro_news_count", 0) > 0
    news_quality = np.mean([
        company_news.get("company_news_confidence", 0.0),
        macro_news.get("macro_confidence_score", 0.0),
    ]) if (has_company or has_macro) else 0.3

    freshness = 1.0 if price_age_bars <= 1 else max(0.0, 1.0 - 0.1 * price_age_bars)

    score = (
        0.45 * completeness
        + 0.20 * news_quality
        + 0.15 * (1.0 if fundamentals_available else 0.0)
        + 0.20 * freshness
    )
    missing_flag = completeness < 0.8
    return float(np.clip(score, 0, 1)), bool(missing_flag)


def build_live_feature_row(
    collected: dict[str, Any],
    ticker: str,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    """Build one clean feature row for the latest timestamp (decision-time).

    ``collected`` is the dict returned by ``pipeline.collect_data``.
    """
    as_of = as_of or now_utc()
    prices = collected.get("prices")
    if prices is None or prices.empty:
        raise ValueError("Cannot build features: no price data collected.")

    context = collected.get("context", {})
    table = build_feature_table(prices, context, ticker)
    last_ts = table.index.max()
    row: dict[str, Any] = table.loc[last_ts].to_dict()

    # --- Live news overlay aggregates ---
    company = collected.get("company", {}) or {}
    macro = collected.get("macro", {}) or {}
    company_agg = aggregate_company_news(company.get("scored", []))
    macro_agg = aggregate_macro_news(macro.get("scored", []))
    row.update(company_agg)
    row.update(macro_agg)
    row["abnormal_news_volume_flag"] = float(company.get("abnormal_news_volume", False))

    # --- Timestamp-safe fundamentals ---
    fundamentals = collected.get("fundamentals", {}) or {}
    fund_feats = compute_fundamental_features(fundamentals, as_of=as_of)
    row.update(fund_feats)

    # --- Data confidence ---
    price_age_bars = int(len(prices) - 1 - prices.index.get_loc(last_ts)) if last_ts in prices.index else 0
    conf, missing = compute_data_confidence_score(
        row, company_agg, macro_agg, bool(fund_feats.get("fundamentals_available")), price_age_bars
    )
    row["data_confidence_score"] = conf
    row["missing_data_flag"] = missing

    # --- Identity / metadata ---
    row["ticker"] = ticker.upper()
    row["timestamp"] = str(last_ts)
    row["prediction_horizon"] = get_settings().get("default_prediction_horizon", "24h")
    row["bar_size"] = get_settings().get("bar_size", "1d")
    row["created_at"] = as_of.isoformat()
    return row
