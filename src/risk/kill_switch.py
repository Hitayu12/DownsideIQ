"""Kill switch + risk-limit gating (spec §14.2, §14.5).

Consults the prediction ledger / paper-trade history to decide whether trading
should be halted (daily loss, weekly drawdown, consecutive losses, pipeline
failure). When the kill switch is active, any SHORT is downgraded to NO TRADE.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils.config_loader import get_risk_limits
from src.utils.logging_utils import get_logger
from src.utils.timestamp_utils import now_utc, to_utc

log = get_logger("risk.kill_switch")


def compute_risk_state(paper_trades: pd.DataFrame | None, risk_cfg: dict | None = None) -> dict[str, Any]:
    """Summarise current risk exposure from realised paper trades."""
    risk_cfg = risk_cfg or get_risk_limits()
    account = float(risk_cfg.get("account_size", 10000.0))
    if paper_trades is None or paper_trades.empty:
        return {"daily_pnl_pct": 0.0, "weekly_pnl_pct": 0.0, "consecutive_losses": 0, "open_positions": 0}

    df = paper_trades.copy()
    df["exit_dt"] = pd.to_datetime(df["exit_time"], utc=True, errors="coerce")
    now = now_utc()
    day_mask = df["exit_dt"] >= to_utc(now.replace(hour=0, minute=0, second=0, microsecond=0))
    week_mask = df["exit_dt"] >= (to_utc(now) - pd.Timedelta(days=7))

    daily_pnl = float(df.loc[day_mask, "pnl"].sum())
    weekly_pnl = float(df.loc[week_mask, "pnl"].sum())

    # Consecutive losses at the tail of the (time-sorted) trade list.
    results = df.sort_values("exit_dt")["result"].tolist()
    consec = 0
    for r in reversed(results):
        if r == "loss":
            consec += 1
        else:
            break
    return {
        "daily_pnl_pct": daily_pnl / account,
        "weekly_pnl_pct": weekly_pnl / account,
        "consecutive_losses": consec,
        "open_positions": 0,
    }


def evaluate_kill_switch(
    paper_trades: pd.DataFrame | None,
    risk_cfg: dict | None = None,
    pipeline_ok: bool = True,
    model_drift: bool = False,
) -> dict[str, Any]:
    """Return {'active': bool, 'reasons': [...], 'state': {...}}."""
    risk_cfg = risk_cfg or get_risk_limits()
    state = compute_risk_state(paper_trades, risk_cfg)
    reasons: list[str] = []

    if state["daily_pnl_pct"] <= -float(risk_cfg.get("max_daily_loss", 0.02)):
        reasons.append(f"daily loss limit hit ({state['daily_pnl_pct']:.2%})")
    if state["weekly_pnl_pct"] <= -float(risk_cfg.get("max_weekly_drawdown", 0.05)):
        reasons.append(f"weekly drawdown limit hit ({state['weekly_pnl_pct']:.2%})")
    if state["consecutive_losses"] >= int(risk_cfg.get("max_consecutive_losses", 2)):
        reasons.append(f"consecutive-loss cooldown ({state['consecutive_losses']})")
    if risk_cfg.get("no_trade_on_pipeline_failure", True) and not pipeline_ok:
        reasons.append("data pipeline failure")
    if risk_cfg.get("no_trade_on_model_drift", True) and model_drift:
        reasons.append("model drift warning active")

    return {"active": bool(reasons), "reasons": reasons, "state": state}


def apply_risk_controls(decision: dict[str, Any], paper_trades: pd.DataFrame | None = None,
                        pipeline_ok: bool = True, model_drift: bool = False) -> dict[str, Any]:
    """Downgrade a SHORT to NO TRADE if the kill switch is active (spec §23)."""
    ks = evaluate_kill_switch(paper_trades, pipeline_ok=pipeline_ok, model_drift=model_drift)
    decision = {**decision, "kill_switch": ks}
    if ks["active"] and decision.get("decision") == "SHORT":
        decision["decision"] = "NO TRADE"
        decision["reasons"] = list(decision.get("reasons", [])) + [f"kill switch: {', '.join(ks['reasons'])}"]
        log.warning("Kill switch downgraded SHORT -> NO TRADE: %s", ks["reasons"])
    return decision
