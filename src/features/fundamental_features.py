"""Fundamental features (spec §8.4 → §9), with strict timestamp gating.

Quarterly fundamentals may only be used for a prediction made AFTER their
public release. ``compute_fundamental_features`` therefore drops earnings-tied
metrics whose ``last_reported_earnings_date`` is on/after the prediction
``as_of``. Slower-moving snapshot metrics (valuation, margins) are passed
through for the live row. Missing fields degrade to NaN, never raise.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from src.utils.config_loader import get_risk_limits
from src.utils.logging_utils import get_logger
from src.utils.timestamp_utils import now_utc, to_utc

log = get_logger("features.fundamentals")


def _f(x: Any) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _released_before(date_str: str | None, as_of: datetime) -> bool:
    """True if a quarterly figure dated ``date_str`` was public before ``as_of``."""
    if not date_str:
        return False
    try:
        return to_utc(datetime.fromisoformat(str(date_str))) < to_utc(as_of)
    except ValueError:
        return False


def compute_fundamental_features(
    fundamentals: dict[str, Any],
    as_of: datetime | None = None,
) -> dict[str, float]:
    """Build the fundamental feature dict for a single prediction timestamp."""
    as_of = as_of or now_utc()
    overview = fundamentals.get("overview", {}) or {}
    earnings = fundamentals.get("earnings", {}) or {}
    basics = fundamentals.get("basic_financials", {}) or {}

    out: dict[str, float] = {}

    # --- Snapshot (slower-moving) metrics ---
    out["valuation_multiple"] = _f(overview.get("PERatio")) or _f(basics.get("peTTM"))
    out["revenue_growth"] = _f(overview.get("QuarterlyRevenueGrowthYOY")) \
        if _f(overview.get("QuarterlyRevenueGrowthYOY")) == _f(overview.get("QuarterlyRevenueGrowthYOY")) \
        else _f(basics.get("revenueGrowthTTMYoy"))
    out["gross_margin"] = _f(basics.get("grossMarginTTM"))
    out["operating_margin"] = _f(overview.get("OperatingMarginTTM")) or _f(basics.get("operatingMarginTTM"))
    out["net_margin"] = _f(overview.get("ProfitMargin")) or _f(basics.get("netProfitMarginTTM"))
    out["debt_to_equity"] = _f(basics.get("totalDebt/totalEquityQuarterly"))
    out["current_ratio"] = _f(basics.get("currentRatioQuarterly"))
    out["beta"] = _f(overview.get("Beta"))

    # --- Earnings-tied metrics: ONLY if released before as_of (timestamp rule) ---
    last_reported = fundamentals.get("last_reported_earnings_date")
    if _released_before(last_reported, as_of):
        out["eps_surprise"] = _f(earnings.get("eps_surprise"))
        out["eps_surprise_pct"] = _f(earnings.get("eps_surprise_pct"))
    else:
        out["eps_surprise"] = np.nan
        out["eps_surprise_pct"] = np.nan
        if last_reported:
            log.info("Suppressing earnings metrics: last_reported %s not before as_of %s.",
                     last_reported, as_of)

    # --- Earnings proximity ---
    dist = fundamentals.get("earnings_date_distance_days")
    out["earnings_date_distance_days"] = _f(dist)
    blackout = int(get_risk_limits().get("earnings_blackout_days", 1))
    out["earnings_risk_flag"] = float(
        dist is not None and 0 <= float(dist) <= max(blackout, 5)
    )

    # --- Composite risk scores (higher = riskier; NaN-safe) ---
    out["balance_sheet_risk_score"] = _balance_sheet_risk(out)
    out["profitability_trend_score"] = _profitability_score(out)
    out["fundamental_risk_score"] = _fundamental_risk(out)
    out["fundamentals_available"] = float(bool(overview or basics))
    return out


def _balance_sheet_risk(f: dict[str, float]) -> float:
    """0 (healthy) .. 1 (risky) from leverage + liquidity."""
    risk, n = 0.0, 0
    de = f.get("debt_to_equity")
    if de == de:  # not NaN
        risk += min(1.0, max(0.0, de / 200.0)); n += 1   # Finnhub D/E often in %
    cr = f.get("current_ratio")
    if cr == cr:
        risk += min(1.0, max(0.0, (1.5 - cr) / 1.5)); n += 1
    return float(risk / n) if n else np.nan


def _profitability_score(f: dict[str, float]) -> float:
    """0 (weak) .. 1 (strong) from margins."""
    vals = [f.get("net_margin"), f.get("operating_margin"), f.get("gross_margin")]
    vals = [v for v in vals if v == v]
    if not vals:
        return np.nan
    # Margins may be fractions (0..1) or percents; normalise heuristically.
    norm = [v / 100.0 if abs(v) > 1.5 else v for v in vals]
    return float(np.clip(np.mean(norm) + 0.5, 0, 1))


def _fundamental_risk(f: dict[str, float]) -> float:
    """Composite downside-relevant fundamental risk, 0..1."""
    parts = []
    bs = f.get("balance_sheet_risk_score")
    if bs == bs:
        parts.append(bs)
    prof = f.get("profitability_trend_score")
    if prof == prof:
        parts.append(1.0 - prof)
    if f.get("earnings_risk_flag"):
        parts.append(0.6)
    return float(np.clip(np.mean(parts), 0, 1)) if parts else np.nan
