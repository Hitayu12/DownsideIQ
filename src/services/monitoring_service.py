"""Monitoring service (spec §9, §20) — health, drift, data-quality, failures.

Aggregates observability signals from the audit tables so the API/dashboard can
surface system health and model reliability. Read-only.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import func, select

from src.core.logging import get_logger
from src.db import models as m
from src.db.session import get_session
from src.providers import (
    alpha_vantage,
    finnhub_provider,
    gemini_provider,
    tavily_provider,
)
from src.risk.kill_switch import evaluate_kill_switch

log = get_logger("services.monitoring")


class MonitoringService:
    def health(self) -> dict[str, Any]:
        """Provider availability + DB connectivity + last-signal recency."""
        db_ok, last_signal = True, None
        try:
            with get_session() as s:
                last_signal = s.scalar(select(func.max(m.FinalSignal.created_at)))
        except Exception as exc:  # noqa: BLE001
            db_ok = False
            log.warning("health_db_error", error=str(exc)[:160])
        return {
            "status": "ok" if db_ok else "degraded",
            "database": db_ok,
            "providers": {
                "tavily": tavily_provider.available(),
                "gemini": gemini_provider.available(),
                "alpha_vantage": alpha_vantage.available(),
                "finnhub": finnhub_provider.available(),
            },
            "last_signal_at": str(last_signal) if last_signal else None,
        }

    def drift(self, ticker: str | None = None, mode: str = "strict", window: int = 50) -> dict[str, Any]:
        """Rolling accuracy / false-positive / data-quality distribution + kill switch."""
        with get_session() as s:
            q = select(m.FinalSignal).where(m.FinalSignal.mode == mode,
                                            m.FinalSignal.prediction_correct.isnot(None))
            if ticker:
                q = q.where(m.FinalSignal.ticker == ticker.upper())
            sigs = list(s.scalars(q.order_by(m.FinalSignal.ts.desc()).limit(window)))

            trades = s.scalars(select(m.PaperTrade).where(m.PaperTrade.mode == mode)).all()
            dq_rows = s.execute(
                select(m.FinalSignal.data_quality, func.count())
                .where(m.FinalSignal.mode == mode).group_by(m.FinalSignal.data_quality)
            ).all()

        resolved = [bool(x.prediction_correct) for x in sigs]
        rolling_acc = (sum(resolved) / len(resolved)) if resolved else None
        shorts = [x for x in sigs if x.decision == "SHORT"]
        fp_short = (sum(1 for x in shorts if not x.prediction_correct) / len(shorts)) if shorts else None

        import pandas as pd
        df = pd.DataFrame([{"pnl": t.pnl or 0, "result": t.result} for t in trades]) if trades else pd.DataFrame()
        ks = evaluate_kill_switch(df if not df.empty else None)

        return {
            "mode": mode,
            "n_resolved_signals": len(resolved),
            "rolling_accuracy": rolling_acc,
            "false_positive_short_rate": fp_short,
            "data_quality_distribution": {row[0]: row[1] for row in dq_rows},
            "kill_switch": ks,
        }

    def model_status(self, ticker: str | None = None) -> list[dict[str, Any]]:
        with get_session() as s:
            q = select(m.ModelPerformance).order_by(m.ModelPerformance.created_at.desc()).limit(50)
            rows = list(s.scalars(q))
        return [{"model_name": r.model_name, "model_version": r.model_version,
                 "feature_set_version": r.feature_set_version, "metric_name": r.metric_name,
                 "metric_value": r.metric_value, "mode": r.mode,
                 "training_date": str(r.training_date)} for r in rows]
