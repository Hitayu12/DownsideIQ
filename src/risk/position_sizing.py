"""Position sizing + stop/take-profit (spec §14.1, §14.3, §14.4).

Volatility-aware sizing: risk a fixed fraction of the account per trade, sized
so that hitting the stop loses exactly that amount. The base size is then scaled
down by model confidence, agreement, and data confidence — weak/uncertain
signals get smaller positions.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.utils.config_loader import get_risk_limits


def compute_sizing(
    decision: dict[str, Any],
    entry_price: float,
    predicted_volatility: float,
    risk_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return sizing + stop/TP for a (short) signal.

    For a SHORT, the adverse direction is UP, so the stop sits above entry and
    the take-profit below.
    """
    risk_cfg = risk_cfg or get_risk_limits()
    account = float(risk_cfg.get("account_size", 10000.0))
    risk_per_trade = float(risk_cfg.get("max_risk_per_trade", 0.005))
    stop_floor = float(risk_cfg.get("stop_loss_floor_pct", 0.01))
    stop_vol_mult = float(risk_cfg.get("stop_loss_vol_mult", 1.5))
    rr = float(risk_cfg.get("take_profit_rr", 1.5))

    pv = predicted_volatility if predicted_volatility == predicted_volatility else stop_floor
    stop_distance = max(stop_floor, stop_vol_mult * pv)          # fractional
    risk_amount = account * risk_per_trade
    base_notional = risk_amount / stop_distance

    # Confidence-scaled sizing (each factor in [0,1]).
    conf = abs(float(decision.get("adjusted_p_downside", 0.5)) - 0.5) * 2.0
    agreement = float(decision.get("model_agreement_score", 0.5))
    data_conf = float(decision.get("data_confidence_score", 0.5))
    scale = float(np.clip(conf * agreement * data_conf, 0.0, 1.0))
    notional = base_notional * scale
    notional = min(notional, account)                           # no leverage in MVP

    shares = notional / entry_price if entry_price > 0 else 0.0
    stop_loss_price = entry_price * (1 + stop_distance)          # short: stop above
    take_profit_price = entry_price * (1 - stop_distance * rr)   # short: TP below

    return {
        "side": "short",
        "entry_price": float(entry_price),
        "stop_distance": float(stop_distance),
        "stop_loss": float(stop_loss_price),
        "take_profit": float(take_profit_price),
        "risk_amount": float(risk_amount),
        "position_notional": float(notional),
        "position_size": float(shares),
        "size_scale": scale,
    }
