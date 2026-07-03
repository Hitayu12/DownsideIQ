"""Pipeline orchestrator (spec §5) — the production request→audit flow.

Wires the services in order. Callable synchronously (CLI/tests) or from a Celery
task. Every stage persists its audit trail; the final signal is written to the
ledger before any outcome is known.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from src.core.config import get_ticker_config
from src.core.logging import get_logger
from src.core.time import now_utc
from src.services.ensemble_service import EnsembleService
from src.services.feature_service import FeatureService
from src.services.ingestion_service import IngestionService
from src.services.ledger_service import LedgerService
from src.services.model_service import ModelService
from src.services.news_scoring_service import NewsScoringService
from src.services.risk_engine_service import RiskEngineService

log = get_logger("pipeline.orchestrator")


def analyze(ticker: str, *, mode: str = "strict", as_of: datetime | None = None,
            persist: bool = True) -> dict[str, Any]:
    """Run the full analysis pipeline for ``ticker`` and write the signal."""
    ticker = ticker.upper()
    as_of = as_of or now_utc()
    log.info("analyze_start", ticker=ticker, mode=mode)

    ingestion = IngestionService().ingest(ticker, as_of=as_of, persist=persist)
    snapshot = FeatureService().build_snapshot(ingestion, persist=persist)
    council = ModelService().predict(ingestion, snapshot, persist=persist)

    # News overlay (live event-risk).
    news = NewsScoringService()
    cfg = get_ticker_config(ticker)
    company_scores = news.score(ingestion.raw_company_news, ticker, cfg.get("name"), as_of, "company")
    macro_scores = news.score(ingestion.raw_macro_news, ticker, None, as_of, "macro")
    if persist:
        news.persist(company_scores + macro_scores, ticker)
    overlay = news.overlay_features(company_scores, macro_scores)

    model_outputs = EnsembleService().assemble(council, snapshot, overlay)
    model_outputs["horizon"] = snapshot.horizon

    decision = RiskEngineService().evaluate(model_outputs, mode=mode, degraded=ingestion.degraded)

    signal_id = None
    if persist:
        signal_id = LedgerService().write_signal(decision, mode=mode)
    decision["signal_id"] = signal_id
    log.info("analyze_complete", ticker=ticker, mode=mode, decision=decision["decision"],
             signal_id=signal_id, data_quality=decision["data_quality"])
    return decision
