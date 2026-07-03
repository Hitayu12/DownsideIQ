"""Final Decision Engine (spec §11–§13) — the heart of DownsideIQ.

Combines the three Council-2 model outputs with the live news overlay and a set
of quality multipliers into a single decision: SHORT / WATCH / NO TRADE.

Pipeline (matches the agreed hybrid architecture):

    1. base_log_odds   = logit(p_downside)                    # historical model
    2. news_shift      = a1·company_news + a2·macro_news
                         + a3·net_catalyst + a4·pv_confirm     # live overlay
                         - a5·uncertainty_penalty
       (news_shift is hard-capped by max_logodds_shift so news can't override
        the model unless it is strong AND price/volume confirms)
    3. adjusted_p      = sigmoid(base_log_odds + news_shift)
    4. base_risk       = 0.45·adjusted_p + 0.30·garch_dvr + 0.25·tail
    5. adjusted_risk   = base_risk · agreement · data_conf · regime_adj
                         - uncertainty_penalty
    6. decision        = gate(adjusted_p, agreement, data_conf, pv_confirm,
                              expected_edge, earnings, ...)
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.utils.config_loader import get_risk_limits, get_settings
from src.utils.logging_utils import get_logger

log = get_logger("models.ensemble")

# Base ensemble weights (spec §11.1) — starting weights; learnable later.
W_CLASSIFIER, W_GARCH, W_QUANTILE = 0.45, 0.30, 0.25


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _g(row: dict, key: str, default: float = 0.0) -> float:
    v = row.get(key, default)
    try:
        v = float(v)
        return default if (v != v) else v   # NaN -> default
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Component scores
# ---------------------------------------------------------------------------
def price_volume_confirmation(row: dict[str, Any]) -> float:
    """Signed price/volume confirmation of DOWNSIDE in [-1, 1].

    Positive => price action confirms the bearish case (abnormal volume,
    negative momentum, weak sector relative strength, volatility expansion,
    gap down, peer weakness, correlation spike, drawdown). News-only signals
    with no confirmation score ~0 and are gated to WATCH, never SHORT.
    """
    signals = []
    signals.append(np.clip(_g(row, "abnormal_volume_score"), 0, 1))            # volume
    signals.append(np.clip(-_g(row, "momentum_score") * 5, -1, 1))             # neg momentum
    signals.append(np.clip(-_g(row, "relative_strength_vs_sector") * 5, -1, 1))# weak vs sector
    signals.append(np.clip((_g(row, "volatility_ratio", 1.0) - 1.0), -1, 1))   # vol expansion
    signals.append(np.clip(-_g(row, "overnight_gap") * 20, -1, 1))             # gap down
    signals.append(np.clip(-_g(row, "return_vs_peers_1b") * 20, -1, 1))        # peer weakness
    signals.append(np.clip(_g(row, "correlation_spike_score") * 3, 0, 1))      # corr spike
    signals.append(np.clip(-_g(row, "drawdown_from_recent_high") * 5, 0, 1))   # drawdown
    return float(np.clip(np.mean(signals), -1, 1))


def model_agreement(adjusted_p: float, garch_dvr: float, tail_score: float) -> float:
    """1 = three models agree on downside magnitude, 0 = maximal disagreement."""
    votes = np.array([adjusted_p, garch_dvr, tail_score], dtype=float)
    votes = votes[~np.isnan(votes)]
    if len(votes) < 2:
        return 0.5
    return float(np.clip(1.0 - 2.0 * np.std(votes), 0.0, 1.0))


def market_regime_adjustment(row: dict[str, Any]) -> float:
    """Risk multiplier in [0.8, 1.3]; higher when broad/sector stress raises downside."""
    adj = 1.0
    if row.get("market_regime") == "risk_off":
        adj += 0.15
    if _g(row, "sector_selloff_flag") >= 1.0:
        adj += 0.10
    if _g(row, "broad_market_selloff_flag") >= 1.0:
        adj += 0.10
    if _g(row, "vix_level") > 28:
        adj += 0.10
    return float(np.clip(adj, 0.8, 1.3))


def uncertainty_penalty(
    row: dict[str, Any],
    agreement: float,
    data_confidence: float,
    adjusted_p: float,
    quantile_out: dict[str, Any],
    min_p: float,
) -> float:
    """Penalty in [0, 0.3]: grows with disagreement, low confidence, wide tails,
    missing data, sudden vol shifts, and signals that barely clear threshold."""
    parts = []
    parts.append(0.10 * (1.0 - agreement))
    parts.append(0.08 * (1.0 - data_confidence))
    parts.append(0.05 if row.get("missing_data_flag") else 0.0)
    # Wide downside-tail spread => more uncertain severity.
    q5, q10 = quantile_out.get("predicted_5pct_return"), quantile_out.get("predicted_10pct_return")
    if q5 is not None and q10 is not None and not (np.isnan(q5) or np.isnan(q10)):
        parts.append(0.05 * float(np.clip(abs(q10 - q5) / 0.02, 0, 1)))
    # Sudden volatility-regime shift.
    if _g(row, "volatility_ratio", 1.0) > 1.5:
        parts.append(0.05)
    # Signal barely clears the gate threshold.
    if 0 < (adjusted_p - min_p) < 0.03:
        parts.append(0.04)
    return float(np.clip(sum(parts), 0.0, 0.30))


# ---------------------------------------------------------------------------
# Main decision
# ---------------------------------------------------------------------------
def decide(model_outputs: dict[str, Any], gate_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Combine Council-2 outputs + overlay into a final decision dict.

    ``gate_overrides`` temporarily relaxes/tightens the signal-quality gate
    thresholds (used by the clearly-labeled *exploratory* backtest). The live /
    default path passes nothing, so the strict institutional gate applies.
    """
    settings = get_settings()
    risk = {**get_risk_limits(), **(gate_overrides or {})}
    overlay = settings.get("news_overlay", {})
    costs = settings.get("costs", {})

    row = model_outputs["feature_row"]
    p_downside = float(model_outputs["classifier"]["p_downside"])
    garch = model_outputs["garch"]
    quant = model_outputs["quantile"]
    garch_dvr = float(garch.get("downside_volatility_risk", 0.0) or 0.0)
    tail_score = float(quant.get("downside_tail_score", 0.0) or 0.0)

    # --- 1-3. News overlay in log-odds space (capped) ---
    company_news = _g(row, "company_news_risk_score")
    macro_news = _g(row, "macro_risk_score")
    net_catalyst = _g(row, "negative_catalyst_score") - _g(row, "positive_catalyst_score")
    pv_confirm = price_volume_confirmation(row)

    # Provisional agreement/penalty use base p; recomputed after overlay below.
    data_confidence = _g(row, "data_confidence_score", 0.5)
    min_p = float(risk.get("minimum_downside_probability", 0.65))

    raw_shift = (
        float(overlay.get("alpha_company_news", 0.8)) * company_news
        + float(overlay.get("alpha_macro_news", 0.5)) * macro_news
        + float(overlay.get("alpha_event_impact", 0.7)) * net_catalyst
        + float(overlay.get("alpha_price_volume_confirmation", 0.6)) * pv_confirm
    )
    cap = float(overlay.get("max_logodds_shift", 1.2))
    news_shift = float(np.clip(raw_shift, -cap, cap)) if overlay.get("enabled", True) else 0.0
    adjusted_p = _sigmoid(_logit(p_downside) + news_shift)

    # --- Quality scores (use adjusted_p) ---
    agreement = model_agreement(adjusted_p, garch_dvr, tail_score)
    regime_adj = market_regime_adjustment(row)
    penalty = uncertainty_penalty(row, agreement, data_confidence, adjusted_p, quant, min_p)

    base_risk = W_CLASSIFIER * adjusted_p + W_GARCH * garch_dvr + W_QUANTILE * tail_score
    adjusted_risk = float(np.clip(base_risk * agreement * data_confidence * regime_adj - penalty, 0, 1))

    # --- Expected edge vs transaction cost (spec §12/§13) ---
    q5 = quant.get("predicted_5pct_return")
    expected_move = adjusted_p * abs(q5) if (q5 is not None and not np.isnan(q5)) else adjusted_p * garch.get("var_estimate", 0.0)
    round_trip_cost = 2.0 * (float(costs.get("commission_bps", 1)) + float(costs.get("slippage_bps", 5))) / 1e4
    expected_edge_bps = (expected_move - round_trip_cost) * 1e4
    min_edge_bps = float(costs.get("min_edge_bps", 20))

    # --- Gate + decision ---
    decision, reasons = _gate(
        adjusted_p=adjusted_p, agreement=agreement, data_confidence=data_confidence,
        pv_confirm=pv_confirm, expected_edge_bps=expected_edge_bps, min_edge_bps=min_edge_bps,
        adjusted_risk=adjusted_risk, row=row, risk=risk, company_news=company_news,
        macro_news=macro_news,
    )

    drivers = _top_drivers(model_outputs, adjusted_p, pv_confirm, company_news, macro_news, garch_dvr, tail_score)

    return {
        "ticker": model_outputs.get("ticker"),
        "timestamp": model_outputs.get("timestamp"),
        "decision": decision,
        "reasons": reasons,
        "p_downside": p_downside,
        "adjusted_p_downside": adjusted_p,
        "news_shift_logodds": news_shift,
        "news_shift_capped": abs(raw_shift) > cap,
        "base_downside_risk_score": float(base_risk),
        "adjusted_downside_risk_score": adjusted_risk,
        "model_agreement_score": agreement,
        "data_confidence_score": data_confidence,
        "market_regime_adjustment": regime_adj,
        "uncertainty_penalty": penalty,
        "price_volume_confirmation": pv_confirm,
        "expected_edge_bps": float(expected_edge_bps),
        "garch": garch,
        "quantile": quant,
        "top_drivers": drivers,
        "news_catalysts": {
            "company_news_risk_score": company_news,
            "macro_risk_score": macro_news,
            "negative_catalyst_score": _g(row, "negative_catalyst_score"),
            "abnormal_news_volume": bool(_g(row, "abnormal_news_volume_flag")),
        },
        "market_regime": row.get("market_regime"),
    }


def _gate(*, adjusted_p, agreement, data_confidence, pv_confirm, expected_edge_bps,
          min_edge_bps, adjusted_risk, row, risk, company_news, macro_news) -> tuple[str, list[str]]:
    """Signal-quality gate → SHORT / WATCH / NO TRADE with human-readable reasons."""
    min_p = float(risk.get("minimum_downside_probability", 0.65))
    min_agree = float(risk.get("minimum_model_agreement", 0.70))
    min_dc = float(risk.get("minimum_data_confidence", 0.75))
    require_pv = bool(risk.get("require_price_volume_confirmation", True))
    pv_threshold = float(risk.get("pv_confirm_threshold", 0.10))

    reasons: list[str] = []

    # Hard blockers -> NO TRADE.
    if row.get("missing_data_flag"):
        return "NO TRADE", ["data pipeline incomplete (missing_data_flag)"]
    if data_confidence < 0.5:
        return "NO TRADE", [f"data confidence too low ({data_confidence:.2f})"]
    dist = row.get("earnings_date_distance_days")
    blackout = int(risk.get("earnings_blackout_days", 1))
    earnings_block = (
        bool(risk.get("block_trade_near_earnings", True))
        and dist is not None and not (isinstance(dist, float) and np.isnan(dist))
        and 0 <= float(dist) <= blackout
    )

    # Strong-SHORT conditions.
    short_ok = (
        adjusted_p >= min_p
        and agreement >= min_agree
        and data_confidence >= min_dc
        and expected_edge_bps >= min_edge_bps
        and (pv_confirm > pv_threshold if require_pv else True)
        and not earnings_block
    )
    if short_ok:
        reasons.append(f"p_downside {adjusted_p:.2f} ≥ {min_p}")
        reasons.append(f"agreement {agreement:.2f} ≥ {min_agree}")
        reasons.append(f"price/volume confirms (pv={pv_confirm:.2f})")
        reasons.append(f"edge {expected_edge_bps:.0f}bps ≥ {min_edge_bps}bps")
        return "SHORT", reasons

    # WATCH: interesting downside signal but not fully confirmed.
    interesting = (
        adjusted_p >= (min_p - 0.10)
        or company_news >= 0.3 or macro_news >= 0.3
        or adjusted_risk >= 0.45
    )
    if interesting:
        if adjusted_p < min_p:
            reasons.append(f"p_downside {adjusted_p:.2f} below {min_p}")
        if agreement < min_agree:
            reasons.append(f"model agreement {agreement:.2f} below {min_agree}")
        if require_pv and pv_confirm <= pv_threshold:
            reasons.append("price/volume not yet confirming news")
        if expected_edge_bps < min_edge_bps:
            reasons.append(f"edge {expected_edge_bps:.0f}bps below {min_edge_bps}bps")
        if earnings_block:
            reasons.append("earnings blackout active")
        return "WATCH", reasons or ["signal present but not strong enough"]

    # Otherwise NO TRADE.
    reasons.append(f"downside signal weak (p={adjusted_p:.2f}, risk={adjusted_risk:.2f})")
    if expected_edge_bps < min_edge_bps:
        reasons.append(f"expected edge {expected_edge_bps:.0f}bps below cost threshold")
    return "NO TRADE", reasons


def _top_drivers(model_outputs, adjusted_p, pv_confirm, company_news, macro_news,
                 garch_dvr, tail_score) -> list[dict[str, Any]]:
    """Rank the strongest contributors to the decision for explainability."""
    candidates = [
        {"driver": "model_downside_probability", "value": round(adjusted_p, 3), "weight": round(W_CLASSIFIER * adjusted_p, 3)},
        {"driver": "garch_downside_volatility", "value": round(garch_dvr, 3), "weight": round(W_GARCH * garch_dvr, 3)},
        {"driver": "quantile_downside_tail", "value": round(tail_score, 3), "weight": round(W_QUANTILE * tail_score, 3)},
        {"driver": "company_news_risk", "value": round(company_news, 3), "weight": round(abs(company_news), 3)},
        {"driver": "macro_news_risk", "value": round(macro_news, 3), "weight": round(abs(macro_news), 3)},
        {"driver": "price_volume_confirmation", "value": round(pv_confirm, 3), "weight": round(abs(pv_confirm), 3)},
    ]
    # Blend in top XGBoost features for transparency.
    for name, imp in (model_outputs.get("classifier", {}).get("top_features", []) or [])[:3]:
        candidates.append({"driver": f"feature:{name}", "value": round(float(imp), 3), "weight": round(float(imp), 3)})
    return sorted(candidates, key=lambda d: d["weight"], reverse=True)[:5]
