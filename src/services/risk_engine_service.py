"""Risk engine service (spec §12, §13, §14) — gates, modes, limits, kill switch.

Consumes the ensemble-scored decision and applies:
  - the signal-quality gate in the ACTIVE MODE (strict=fixed thresholds,
    research=percentile-calibrated against trailing history),
  - risk limits + kill switch (downgrades SHORT -> NO TRADE when tripped),
  - position sizing for actionable signals,
and emits the full governance record (why SHORT/WATCH/NO TRADE).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.core.config import get_thresholds
from src.core.logging import get_logger
from src.db import repositories as repo
from src.db.session import get_session
from src.models.ensemble_engine import decide as ensemble_decide
from src.risk.kill_switch import evaluate_kill_switch
from src.risk.position_sizing import compute_sizing

log = get_logger("services.risk_engine")

_STRICT_GATE_KEYS = (
    "minimum_downside_probability", "minimum_model_agreement",
    "minimum_data_confidence", "require_price_volume_confirmation", "pv_confirm_threshold",
)


class RiskEngineService:
    def evaluate(self, model_outputs: dict[str, Any], *, mode: str = "strict",
                 degraded=None, pipeline_ok: bool = True) -> dict[str, Any]:
        thresholds = get_thresholds()
        mode = mode if mode in thresholds.get("modes", {}) else thresholds.get("default_mode", "strict")
        mcfg = thresholds["modes"][mode]
        ticker = model_outputs.get("ticker")

        overrides = self._gate_overrides(mcfg, mode)
        decision = ensemble_decide(model_outputs, gate_overrides=overrides)

        if mode == "research":
            decision = self._apply_percentile_gate(decision, mcfg, ticker)

        # --- kill switch + risk limits ---
        trades = self._load_trades_df(ticker, mode)
        ks = evaluate_kill_switch(trades, pipeline_ok=pipeline_ok)
        if ks["active"] and decision["decision"] == "SHORT":
            decision["decision"] = "NO TRADE"
            decision.setdefault("reasons", []).append(f"kill switch: {', '.join(ks['reasons'])}")
            self._record_risk_event(ticker, "kill_switch", ks["reasons"])

        # --- position sizing for actionable signals ---
        sizing = None
        if decision["decision"] == "SHORT":
            entry = model_outputs.get("feature_row", {}).get("current_price") or 0.0
            sizing = compute_sizing(decision, entry, decision["garch"]["forecast_volatility"])

        decision["governance"] = self._governance(decision, mode, mcfg, ks, degraded, sizing)
        decision["mode"] = mode
        decision["kill_switch_active"] = ks["active"]
        decision["data_quality"] = degraded.status.value if degraded else "ok"
        decision["position_sizing"] = sizing
        log.info("risk_evaluated", ticker=ticker, mode=mode, decision=decision["decision"],
                 kill_switch=ks["active"], data_quality=decision["data_quality"])
        return decision

    # ---- gate construction ----
    def _gate_overrides(self, mcfg: dict, mode: str) -> dict[str, Any]:
        if mode == "research":
            # Relax the fixed gate so the percentile gate governs the decision.
            return {
                "minimum_downside_probability": 0.0,
                "minimum_model_agreement": 0.0,
                "minimum_data_confidence": mcfg.get("minimum_data_confidence", 0.55),
                "require_price_volume_confirmation": mcfg.get("require_price_volume_confirmation", True),
                "pv_confirm_threshold": mcfg.get("pv_confirm_threshold", 0.0),
            }
        return {k: mcfg[k] for k in _STRICT_GATE_KEYS if k in mcfg}

    def _apply_percentile_gate(self, decision: dict, mcfg: dict, ticker: str) -> dict:
        """Research mode: SHORT if adjusted risk is in the top percentile of history."""
        risk = decision["adjusted_downside_risk_score"]
        hist = self._historical_risk(ticker, int(mcfg.get("trailing_window", 120)))
        if len(hist) < 20:
            return decision   # not enough history; keep relaxed-fixed result
        short_cut = float(np.quantile(hist, mcfg.get("short_percentile", 0.85)))
        watch_cut = float(np.quantile(hist, mcfg.get("watch_percentile", 0.60)))
        if risk >= short_cut and decision["decision"] != "NO TRADE":
            decision["decision"] = "SHORT"
            decision["reasons"] = [f"research: adj risk {risk:.2f} ≥ p{int(mcfg['short_percentile']*100)} ({short_cut:.2f})"]
        elif risk >= watch_cut:
            decision["decision"] = "WATCH"
            decision["reasons"] = [f"research: adj risk {risk:.2f} ≥ p{int(mcfg['watch_percentile']*100)} ({watch_cut:.2f})"]
        else:
            decision["decision"] = "NO TRADE"
            decision["reasons"] = [f"research: adj risk {risk:.2f} below watch percentile"]
        return decision

    def _historical_risk(self, ticker: str, window: int) -> list[float]:
        with get_session() as s:
            sigs = repo.signal_history(s, ticker=ticker, mode="research", limit=window)
        return [s.adjusted_downside_risk_score for s in sigs if s.adjusted_downside_risk_score is not None]

    # ---- limits / kill switch ----
    def _load_trades_df(self, ticker: str, mode: str) -> pd.DataFrame | None:
        with get_session() as s:
            trades = repo.paper_trades(s, ticker=ticker, mode=mode)
        if not trades:
            return None
        return pd.DataFrame([{"exit_time": t.exit_time, "pnl": t.pnl or 0.0,
                              "result": t.result or "flat"} for t in trades if t.exit_time])

    def _record_risk_event(self, ticker, event_type, reasons):
        with get_session() as s:
            repo.log_risk_event(s, event_type, ticker=ticker, severity="warning",
                                detail={"reasons": reasons})

    # ---- governance (§12) ----
    def _governance(self, decision, mode, mcfg, ks, degraded, sizing) -> dict[str, Any]:
        passed = decision["decision"] in ("SHORT",)
        return {
            "decision": decision["decision"],
            "mode": mode,
            "threshold_mode": mcfg.get("type"),
            "reason": "; ".join(decision.get("reasons", [])),
            "scores": {
                "p_downside": decision.get("p_downside"),
                "adjusted_p_downside": decision.get("adjusted_p_downside"),
                "adjusted_downside_risk_score": decision.get("adjusted_downside_risk_score"),
                "model_agreement_score": decision.get("model_agreement_score"),
                "data_confidence_score": decision.get("data_confidence_score"),
                "price_volume_confirmation": decision.get("price_volume_confirmation"),
                "expected_edge_bps": decision.get("expected_edge_bps"),
            },
            "news_catalysts": decision.get("news_catalysts"),
            "gates": {
                "min_probability": mcfg.get("minimum_downside_probability"),
                "min_agreement": mcfg.get("minimum_model_agreement"),
                "min_data_confidence": mcfg.get("minimum_data_confidence"),
                "require_pv_confirmation": mcfg.get("require_price_volume_confirmation"),
                "gate_passed_for_short": passed,
            },
            "kill_switch": ks,
            "data_quality": {
                "status": degraded.status.value if degraded else "ok",
                "reasons": degraded.reasons if degraded else [],
                "news_confidence_multiplier": degraded.news_confidence_multiplier if degraded else 1.0,
            },
            "position_sizing": sizing,
            "top_drivers": decision.get("top_drivers"),
        }
