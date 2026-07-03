"""Prediction ledger + signal governance service (spec §12, §15, §34).

Writes the full, auditable signal record BEFORE the outcome is known, and later
reconciles it against the realised next-session return (+ attribution). The
``final_signals`` table is the canonical audit ledger.
"""
from __future__ import annotations

import uuid
from typing import Any

from src.core.logging import get_logger
from src.core.time import now_utc, to_utc
from src.db import repositories as repo
from src.db.session import get_session

log = get_logger("services.ledger")


class LedgerService:
    def write_signal(self, decision: dict[str, Any], *, mode: str,
                     feature_snapshot_id: int | None = None) -> str:
        """Persist a final signal (governance record). Returns signal_id."""
        signal_id = uuid.uuid4().hex[:16]
        row = {
            "signal_id": signal_id,
            "ticker": decision.get("ticker"),
            "ts": to_utc(decision.get("timestamp")).to_pydatetime(),
            "horizon": decision.get("horizon", "24h"),
            "mode": mode,
            "decision": decision["decision"],
            "p_downside": decision.get("p_downside"),
            "adjusted_p_downside": decision.get("adjusted_p_downside"),
            "base_downside_risk_score": decision.get("base_downside_risk_score"),
            "adjusted_downside_risk_score": decision.get("adjusted_downside_risk_score"),
            "model_agreement_score": decision.get("model_agreement_score"),
            "data_confidence_score": decision.get("data_confidence_score"),
            "price_volume_confirmation": decision.get("price_volume_confirmation"),
            "news_shift_logodds": decision.get("news_shift_logodds"),
            "expected_edge_bps": decision.get("expected_edge_bps"),
            "uncertainty_penalty": decision.get("uncertainty_penalty"),
            "reason": "; ".join(decision.get("reasons", [])),
            "governance": decision.get("governance", {}),
            "data_quality": decision.get("data_quality", "ok"),
            "kill_switch_active": bool(decision.get("kill_switch_active", False)),
            "feature_snapshot_id": feature_snapshot_id,
        }
        with get_session() as s:
            repo.save_signal(s, row)
        log.info("signal_logged", signal_id=signal_id, ticker=row["ticker"],
                 decision=row["decision"], mode=mode, data_quality=row["data_quality"])
        return signal_id

    def get(self, signal_id: str) -> dict[str, Any] | None:
        with get_session() as s:
            sig = repo.get_signal(s, signal_id)
            if not sig:
                return None
            return _signal_to_dict(sig)

    def latest(self, ticker: str | None = None, mode: str = "strict") -> dict[str, Any] | None:
        with get_session() as s:
            sig = repo.latest_signal(s, ticker=ticker, mode=mode)
            return _signal_to_dict(sig) if sig else None

    def history(self, ticker: str | None = None, mode: str = "strict", limit: int = 200) -> list[dict]:
        with get_session() as s:
            return [_signal_to_dict(x) for x in repo.signal_history(s, ticker=ticker, mode=mode, limit=limit)]

    def update_outcomes(self, prices_by_ticker: dict[str, Any], horizon_bars: int = 1) -> int:
        """Fill realised returns for matured signals across tickers. Returns count updated."""
        import pandas as pd

        updated = 0
        with get_session() as s:
            open_sigs = repo.open_signals_without_outcome(s)
            for sig in open_sigs:
                prices = prices_by_ticker.get(sig.ticker)
                if prices is None or prices.empty:
                    continue
                idx = prices.sort_index().index
                ts = to_utc(sig.ts)
                after = idx[idx > ts]
                if len(after) < horizon_bars:
                    continue
                entry = float(prices.loc[idx[idx <= ts]]["close"].iloc[-1]) if (idx <= ts).any() else None
                if entry is None:
                    continue
                future_close = float(prices.loc[after[horizon_bars - 1]]["close"])
                ret = (future_close - entry) / entry
                correct = (ret < 0) if sig.decision in ("SHORT", "WATCH") else (ret >= -0.005)
                repo.update_signal_outcome(s, sig.signal_id, ret, bool(correct))
                updated += 1
        log.info("outcomes_updated", count=updated)
        return updated


def _signal_to_dict(sig) -> dict[str, Any]:
    return {
        "signal_id": sig.signal_id, "ticker": sig.ticker, "timestamp": str(sig.ts),
        "horizon": sig.horizon, "mode": sig.mode, "decision": sig.decision,
        "p_downside": sig.p_downside, "adjusted_p_downside": sig.adjusted_p_downside,
        "adjusted_downside_risk_score": sig.adjusted_downside_risk_score,
        "model_agreement_score": sig.model_agreement_score,
        "data_confidence_score": sig.data_confidence_score,
        "price_volume_confirmation": sig.price_volume_confirmation,
        "expected_edge_bps": sig.expected_edge_bps, "reason": sig.reason,
        "governance": sig.governance, "data_quality": sig.data_quality,
        "kill_switch_active": sig.kill_switch_active,
        "actual_return_24h": sig.actual_return_24h, "prediction_correct": sig.prediction_correct,
        "created_at": str(sig.created_at),
    }
