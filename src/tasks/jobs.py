"""Celery task definitions — the async units the API enqueues.

Each task is a thin wrapper around a service/orchestrator call so the same logic
is unit-testable synchronously (call ``.run(...)`` or the orchestrator directly).
"""
from __future__ import annotations

from typing import Any

from src.core.logging import get_logger
from src.tasks.celery_app import celery_app

log = get_logger("tasks")


@celery_app.task(name="analyze", bind=True)
def analyze_task(self, ticker: str, mode: str = "strict") -> dict[str, Any]:
    from src.pipeline.orchestrator import analyze

    log.info("task_analyze_start", ticker=ticker, mode=mode, task_id=self.request.id)
    decision = analyze(ticker, mode=mode)
    return {
        "signal_id": decision.get("signal_id"), "ticker": decision.get("ticker"),
        "decision": decision["decision"], "mode": mode,
        "adjusted_p_downside": decision.get("adjusted_p_downside"),
        "adjusted_downside_risk_score": decision.get("adjusted_downside_risk_score"),
        "data_quality": decision.get("data_quality"), "reason": decision["governance"]["reason"],
    }


@celery_app.task(name="train", bind=True)
def train_task(self, ticker: str) -> dict[str, Any]:
    from src.services.ingestion_service import IngestionService
    from src.services.model_service import ModelService

    ingestion = IngestionService().ingest(ticker)
    metrics = ModelService().train(ingestion)
    return {"ticker": ticker, "metrics": metrics}


@celery_app.task(name="update_outcomes", bind=True)
def update_outcomes_task(self, tickers: list[str] | None = None) -> dict[str, Any]:
    from src.providers.prices import fetch_ohlcv
    from src.services.ledger_service import LedgerService

    led = LedgerService()
    tickers = tickers or _distinct_open_tickers()
    prices = {t: fetch_ohlcv(t, require=False) for t in tickers}
    return {"updated": led.update_outcomes(prices)}


def _distinct_open_tickers() -> list[str]:
    from sqlalchemy import select
    from src.db import models as m
    from src.db.session import get_session

    with get_session() as s:
        rows = s.scalars(select(m.FinalSignal.ticker).where(
            m.FinalSignal.actual_return_24h.is_(None)).distinct())
        return list(rows)
