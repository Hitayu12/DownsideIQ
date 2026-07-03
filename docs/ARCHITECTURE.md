# DownsideIQ — Architecture

A **modular monolith**: clean in-process services with typed interfaces, one
FastAPI backend, one Celery worker, one Streamlit frontend. Production-grade
boundaries without distributed-systems overhead; any service can later be split
out behind its existing interface.

## Layers

```
            ┌──────────────┐      ┌──────────────┐
            │  Streamlit   │ HTTP │   FastAPI    │   enqueue   ┌─────────┐
            │  dashboard   │─────▶│   backend    │────────────▶│  Redis  │
            │ (API client) │◀─────│ (/analyze…)  │◀────────────│  broker │
            └──────────────┘ JSON └──────┬───────┘   poll      └────┬────┘
                                         │                          │
                                         ▼                          ▼
                                  ┌──────────────────────────────────────┐
                                  │            Celery worker             │
                                  │   pipeline.orchestrator.analyze()    │
                                  └──────────────────┬───────────────────┘
                                                     ▼
   ingestion → feature → model → news_scoring → ensemble → risk_engine → ledger
        │          │        │          │            │           │          │
        └──────────┴────────┴──────────┴────────────┴───────────┴──────────┘
                                   persist (SQLAlchemy)
                                          ▼
   raw_price_data · raw_news_results · structured_news_scores · feature_snapshots
   · model_predictions · final_signals · paper_trades · post_trade_attribution
   · risk_events · model_performance · system_logs
```

## Packages (`src/`)
- **core/** — config (pydantic-settings + 5 YAML), structured JSON logging,
  typed errors + `DegradedMode`, tz clock + leakage guards.
- **domain/** — Pydantic contracts exchanged between services.
- **db/** — SQLAlchemy 2.0 ORM (11 tables), session, repositories, Alembic.
- **providers/** — hardened external clients (timeout, retry, fallback).
- **services/** — ingestion, feature, news_scoring, model, ensemble,
  risk_engine, ledger, paper_trading, monitoring.
- **models/** — ML model classes + `ModelRegistry` (versioned artifacts).
- **pipeline/** — orchestrator (the request→audit flow).
- **tasks/** — Celery app + tasks.
- **api/** — FastAPI app + schemas.
- **dashboard/** — Streamlit (calls the API only).
- **cli.py** — Typer commands.

## Key decisions
- **Refactor & preserve**: validated MVP feature math, models, overlay, gate,
  sizing, and tests were moved into services, not rewritten.
- **Async via Celery + Redis**: `/analyze` enqueues a job; clients poll
  `/jobs/{id}`. `CELERY_EAGER=true` runs inline for local dev/tests (no Redis).
- **SQLite → Postgres** by changing `DATABASE_URL` only; Alembic owns migrations.
- **Two modes, never mixed**: `strict` (fixed thresholds) vs `research`
  (percentile-calibrated), separated in config, DB (`mode`), API, dashboard.
