# DownsideIQ — Production Implementation Plan

> Status: **active refactor** from validated MVP → deployable research platform.
> Locked decisions (2026-05-30): **refactor & preserve validated logic · modular
> monolith · SQLAlchemy 2.0 + Alembic · async Celery + Redis job queue · FastAPI
> backend · Streamlit as API client only.**

---

## 1. Architectural principles

1. **Modular monolith.** Each service is an in-process module with a typed
   Pydantic interface and a single responsibility. One deployable backend
   (FastAPI) + one worker (Celery) + one frontend (Streamlit). No
   distributed-systems overhead, but clean boundaries that *could* be split out.
2. **Typed contracts.** Services exchange Pydantic domain models
   (`src/domain/`), never loose dicts. Inputs/outputs are validated at the edges.
3. **Everything auditable.** Every external fetch, feature snapshot, model
   output, decision, trade, risk event, and error is timestamped and persisted.
4. **Fail loud on data, soft on enrichment.** Missing *price* data blocks
   signals (hard error). Missing *news/fundamentals* degrades gracefully with an
   explicit `data_quality` status — never silent, never fake confidence.
5. **Leakage discipline preserved.** Backward-looking features; the only future
   data touched is the training/eval label, after the prediction is logged.
6. **Two modes, never mixed.** `strict` (capital/live discipline, fixed
   thresholds) vs `research` (percentile-calibrated, exploratory). Separated in
   config, DB (`mode` column), API, and dashboard.

---

## 2. Target layout

```
src/
├── core/        config (pydantic-settings, 5 yaml + .env, startup validation),
│                structured JSON logging, typed errors, tz clock + leakage guards
├── domain/      Pydantic contracts: Market, NewsItem/NewsScore, FeatureSnapshot,
│                ModelPrediction, Signal, PaperTrade, RiskStatus, Attribution
├── db/          SQLAlchemy 2.0 ORM (11 tables), session (SQLite→Postgres via URL),
│                repositories (per-aggregate DAO), Alembic migrations
├── providers/   yfinance, tavily, gemini, alpha_vantage, finnhub — each with
│                timeout + tenacity retry + structured error + graceful fallback
├── services/    ingestion, feature, news_scoring, model, ensemble, risk_engine,
│                signal, ledger, paper_trading, monitoring
├── models/      ML classes (downside/volatility/quantile) + ModelRegistry
├── pipeline/    orchestrator wiring the production flow (sync core, callable by Celery)
├── tasks/       Celery app + task definitions (analyze, train, update-outcomes, backtest)
├── api/         FastAPI app, routers, response schemas, dependencies
├── dashboard/   Streamlit — calls the API only
└── cli.py       Typer/argparse CLI (7 commands)
config/   settings.yaml · risk_limits.yaml · model_config.yaml · data_sources.yaml · thresholds.yaml
docs/     ARCHITECTURE · DATA_PIPELINE · MODELING · RISK_ENGINE · API · IMPLEMENTATION_PLAN
deploy/   Dockerfile · docker-compose.yml (api · worker · redis · dashboard · [postgres profile])
tests/    unit/ + integration/ (mocks for interfaces & failure modes only)
```

### Reuse map (validated MVP → services)
| Existing (`src/…`) | Becomes |
|---|---|
| `features/*` | `services/feature_service.py` (logic preserved) |
| `news/*` | `services/news_scoring_service.py` + `providers/{tavily,gemini}.py` |
| `models/{downside,volatility,quantile}` | `models/` + `services/model_service.py` |
| `models/ensemble_engine.py` | `services/ensemble_service.py` + `services/signal_service.py` |
| `risk/*`, `kill_switch` | `services/risk_engine_service.py` |
| `trading/{signal_logger,paper_trader}` | `services/{ledger,paper_trading}_service.py` |
| `backtesting/*` | `services/backtest` + `monitoring_service.py` |
| `utils/{timestamp,config,...}` | `core/` |
| existing 50 tests | kept, moved under `tests/unit`, expanded |

---

## 3. Database schema (SQLAlchemy; SQLite now, Postgres later)

`raw_price_data · raw_news_results · structured_news_scores · feature_snapshots ·
model_predictions · final_signals · paper_trades · post_trade_attribution ·
risk_events · model_performance · system_logs`

Conventions: surrogate PK + natural keys, UTC timestamps, `mode` column on
signal/trade/performance tables, FK from signals→predictions→feature_snapshots
for full lineage, `created_at`/`updated_at` everywhere. Alembic for migrations.

---

## 4. Production flow (request → audit)

```
API POST /analyze/{ticker}  ──>  enqueue Celery job, return job_id
   worker:
     ingestion_service     pull price/news/fundamentals/market ctx  (validate, persist raw)
   → feature_service       build leakage-safe snapshot              (validate; persist)
   → model_service         run registered models (versioned)        (persist predictions)
   → news_scoring_service  score live events (gemini→heuristic)      (persist scores)
   → ensemble_service      adjusted downside risk (capped overlay)
   → risk_engine_service   gates + limits + kill switch + mode
   → signal_service        SHORT / WATCH / NO TRADE + full governance record
   → ledger_service        write audit record (final_signals)
   → paper_trading_service track hypothetical trade (strict) / backtest (research)
   → monitoring_service    drift, data-quality, failure metrics
API GET /signals/{id}  ──>  poll result;  dashboard renders
```

---

## 5. API surface (FastAPI)
`GET /health · POST /analyze/{ticker} (→job) · GET /jobs/{job_id} · GET /signals/latest ·
GET /signals/history · GET /predictions/{signal_id} · GET /risk/status ·
GET /paper-trades · GET /model/status`. Pydantic response schemas; no auth (localhost; roadmap item).

## 6. CLI (Typer)
`collect · train · analyze · update-outcomes · backtest · run-api · run-dashboard` (+ `run-worker`, `migrate`).

## 7. Error handling & observability
Per-provider: timeout, `tenacity` retry (exp backoff) on transient errors,
structured error log, typed fallback, `DataQuality`/`DegradedMode` status surfaced
to the signal. Hard rule: yfinance failure ⇒ block signal + `DataQualityError`;
Tavily ⇒ price-only, `news_confidence=0`; Gemini ⇒ heuristic; AV/Finnhub ⇒ skip
fundamentals. Structured JSON logs for every lifecycle event + `system_logs` table.

## 8. Model registry & governance
Every prediction carries `model_name, model_version, training_date,
feature_set_version, prediction_timestamp`. Artifacts on disk (`models_store/`)
+ metadata row in `model_performance`/registry. Every signal carries the full
governance payload (§12 of the brief): gates passed/failed, threshold mode,
sizing, kill-switch + data-quality status, and a human-readable reason.

---

## 9. Build layers (verified at each step)
1. ✅ **Structure + config + env validation + logging + errors** — `core/{config,logging,errors,time}`, 5 config files, pydantic-settings startup validation, structured JSON logs, utils/ shims keep 50 tests green.
2. ✅ **Database schema** — 11 ORM tables, session (SQLite→Postgres via URL), repositories, Alembic (initial migration `c76f66fba9ea` applied; `render_as_batch` for SQLite).
3. ✅ **Data providers** — `providers/{base,prices,tavily_provider,gemini_provider,alpha_vantage,finnhub_provider}`. Tenacity retry+timeout from `data_sources.yaml`; hard rule (yfinance fail/empty/stale/future → `DataQualityError`, blocks signal); soft fallbacks for news/LLM/fundamentals; Gemini circuit breaker. 10 mock failure-mode tests + live smoke.
4. ✅ **Feature service** (+ ingestion service) — typed `FeatureSnapshot`, reuses validated feature math, data-validation blocks bad data, persists snapshots; ingestion on hardened providers persists raw price/news. Live-verified + unit tests.
5. ✅ **News scoring service** — Gemini (provider, cost-capped, circuit-broken) → heuristic fallback; persists `structured_news_scores`; overlay aggregates. Offline tests.
6. ✅ **Model service + registry** — versioned train/predict (model_name/version/training_date/feature_set_version on every prediction), walk-forward metrics → `model_performance`, artifacts + JSON registry index. Live smoke + registry tests.
7. ✅ **Risk engine** — mode-aware gate (strict fixed / research percentile-calibrated), risk limits + kill switch (downgrades SHORT→NO TRADE), full §12 governance record + position sizing.
8. ✅ **Ensemble + signal + ledger services** — overlay merge; `final_signals` audit ledger (governance JSON, data-quality, kill-switch); outcome reconciliation. End-to-end pipeline orchestrator reproduces validated MVP decision (WATCH, adj_p 0.615) on real data.
9. ✅ **Paper trading + monitoring services** — DB-backed trade sim/perf (mode-tagged); monitoring health/drift/data-quality/model-status.
10. ✅ **Celery tasks + FastAPI backend** — async `/analyze`→`/jobs` (Redis; `CELERY_EAGER` for dev), all 9 endpoints, Typer CLI (collect/train/analyze/update-outcomes/backtest/run-api/run-worker/run-dashboard/migrate).
11. ✅ **Dashboard (API client) + Docker compose** — Streamlit calls API only (5 panels); Dockerfile + compose (api/worker/redis/dashboard, postgres profile).
12. ✅ **Tests + documentation** — 75 tests (unit + integration incl. API/full-pipeline); ARCHITECTURE/DATA_PIPELINE/MODELING/RISK_ENGINE/API docs + production README.

**ALL 12 LAYERS COMPLETE.** 75 tests pass; backend + worker + dashboard + Docker + docs in place.

Each layer: implement → `pytest` green → smoke-verify → update this plan's
checkboxes + memory. Mocks only for interface/failure tests; **no model metric
ever comes from mock data**.

## 10. Deferred / roadmap
API auth, multi-ticker scale-out, true microservice split, MLflow registry,
Postgres in prod, real intraday provider, news meta-model from logged data.
