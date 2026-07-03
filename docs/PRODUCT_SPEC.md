<!-- ============================================================= -->
<!--  DownsideIQ — Product Specification Sheet                      -->
<!--  Document version 1.0 · Status: Released (research preview)    -->
<!-- ============================================================= -->

# DownsideIQ™
## Live Short-Horizon Equity Downside-Risk & Signal Engine
### Product Specification Sheet — v1.0

| | |
|---|---|
| **Product** | DownsideIQ — Equity Downside-Risk & Signal Engine |
| **Category** | Quantitative risk research & decision-support platform |
| **Deployment** | Self-hosted (Docker) · single-node modular monolith |
| **Edition** | Research Preview (paper-trading only) |
| **Document status** | Publishable · v1.0 |
| **Classification** | Not investment advice · live trading disabled by design |

---

## 1. Executive Summary

DownsideIQ is a self-hosted, fully auditable platform that estimates the
**probability and severity of a 12–24 hour downside move** in a selected equity
and converts that estimate into one of three governed, explainable decisions —
**SHORT**, **WATCH**, or **NO TRADE**.

The platform is built on a single operating principle:

> **Prediction is not the edge. Risk-controlled decision-making is the edge.**

Rather than chasing raw price forecasts, DownsideIQ couples a small council of
complementary statistical models with a configurable risk engine, a complete
prediction-audit ledger, and post-trade attribution — so every decision is
timestamped, explained, and reconciled against the realised outcome.

**Primary outcomes for the user**
- A defensible, explainable downside-risk assessment per ticker, on demand.
- Institutional-style discipline: most outputs are *NO TRADE* by design.
- A permanent, queryable audit trail of every prediction and its result.
- A clear, validated upgrade path from research → paper trading → (eventually,
  after extensive validation) supervised real-capital testing.

---

## 2. Positioning & Differentiation

| Generic stock predictor | DownsideIQ |
|---|---|
| Single model, raw price target | Council of 3 risk-specialised models + ensemble |
| Random train/test split | Walk-forward, time-ordered validation only |
| Untimestamped sentiment | Live, timestamped, capped news event-risk overlay |
| Accuracy-only reporting | Downside probability, tail severity, calibration, risk metrics |
| No transaction-cost realism | Cost + slippage gating on every signal |
| "Always-on" signal | No-trade discipline; trades must clear a quality gate |
| Black box | Full per-signal governance record + attribution |
| Notebook / script | Service-oriented, API-first, containerised, tested |

**Target users:** quantitative researchers, risk analysts, systematic-trading
engineers, and finance/ML practitioners who need an auditable downside-risk
workbench and a credible portfolio/interview-grade reference platform.

---

## 3. Core Capabilities

### 3.1 Data Intelligence Council
Gathers, validates, timestamps, and structures market context:
- **Price & volume** (OHLCV) via yfinance — primary, keyless source.
- **Market context** — index ETFs (SPY/QQQ), sector ETFs (SMH/SOXX), peer
  basket, volatility proxy (VIX).
- **Live news** — macro & company event search (Tavily) classified into a
  structured event taxonomy.
- **Fundamentals & earnings** — Alpha Vantage + Finnhub, timestamp-gated so
  unreleased figures can never leak into a prediction.

### 3.2 News Event-Risk Overlay
- Structured scoring of every news item: `event_type`, sentiment, relevance,
  source credibility, recency, company-specificity, expected direction & impact,
  confidence.
- **Hybrid scorer:** Google Gemini (LLM) for the highest-priority items with an
  always-available heuristic/lexicon fallback — identical output schema, graceful
  degradation, cost-capped, circuit-broken.
- Applied as a **capped log-odds adjustment** to the base model probability —
  news can never override the model unless credibility, relevance, **and**
  price/volume confirmation are all strong.

### 3.3 Prediction & Risk Council (three models, three questions)
| Model | Question answered | Output |
|---|---|---|
| XGBoost downside classifier | *How likely is a meaningful drop?* | `p_downside`, feature importances |
| GARCH/EGARCH volatility | *How volatile, and is downside vol elevated?* | forecast vol, VaR, expected shortfall |
| Quantile regression | *If it drops, how bad?* | 5th/10th-percentile return, tail score |

### 3.4 Final Decision Engine
Weighted ensemble → capped news overlay → model-agreement, data-confidence,
market-regime, and uncertainty adjustments → **adjusted downside-risk score**.

### 3.5 Risk Engine & Governance
- **Two modes, never mixed:** *Strict* (fixed institutional thresholds, for
  live/capital discipline) and *Research* (percentile-calibrated, exploratory).
- Signal-quality gate, volatility-aware position sizing, stop-loss / take-profit,
  daily-loss & weekly-drawdown limits, consecutive-loss cooldown, **kill switch**.
- Every signal emits a complete governance record (see §6.3).

### 3.6 Accountability Layer
- **Prediction ledger** — every signal written *before* the outcome is known.
- **Paper-trading engine** — simulated entries/exits, P&L, performance metrics.
- **Post-trade attribution** — decomposes realised moves into market / sector /
  company-specific components with a generated "lesson."
- **Monitoring** — rolling accuracy, false-positive-short rate, data-quality
  distribution, drift, provider health.

### 3.7 Interfaces
- **REST API** (FastAPI) with async job execution.
- **Web terminal** — institutional single-page application.
- **CLI** — operational commands (collect/train/analyze/backtest/…).

---

## 4. Technical Specifications

| Attribute | Specification |
|---|---|
| **Architecture** | Modular monolith; typed in-process services |
| **Language / runtime** | Python 3.11+ |
| **Backend** | FastAPI (ASGI / Uvicorn) |
| **Async execution** | Celery workers, Redis broker/result backend |
| **Persistence** | SQLAlchemy 2.0 ORM; SQLite (dev) → PostgreSQL (prod) |
| **Schema migrations** | Alembic |
| **ML / stats** | scikit-learn, XGBoost, `arch` (GARCH), statsmodels |
| **Frontend** | Vanilla JS SPA + Plotly (served by FastAPI) |
| **Config** | YAML (5 files) + `.env` (pydantic-settings, validated at startup) |
| **Logging** | Structured JSON (structlog) + `system_logs` table |
| **Containerisation** | Docker + docker-compose (api · worker · redis · dashboard; Postgres profile) |
| **Testing** | pytest — 75 unit + integration tests |
| **Default ticker / horizon** | NVDA · 24h next-session (configurable; intraday-ready) |

### 4.1 Service Inventory
`ingestion · feature · news_scoring · model · ensemble · risk_engine ·
ledger · paper_trading · monitoring` — each with a single responsibility and a
typed Pydantic interface.

### 4.2 Data Model (11 audited tables)
`raw_price_data · raw_news_results · structured_news_scores · feature_snapshots ·
model_predictions · final_signals · paper_trades · post_trade_attribution ·
risk_events · model_performance · system_logs`

All records are UTC-timestamped with full lineage
(signal → feature snapshot → model predictions).

### 4.3 Model Governance & Versioning
Every prediction carries `model_name`, `model_version`, `training_date`,
`feature_set_version`, and `prediction_timestamp`. Artifacts are stored in a
versioned registry; walk-forward metrics are persisted to `model_performance`.

---

## 5. Data Sources & Provider Resilience

| Provider | Role | On failure |
|---|---|---|
| yfinance | OHLCV (primary) | **Hard stop** — signal blocked (`DataQualityError`) |
| Tavily | live news search | Price-only mode; `news_confidence = 0` |
| Google Gemini | structured news scoring | Heuristic fallback (circuit breaker) |
| Alpha Vantage | fundamentals / earnings | Fundamentals skipped |
| Finnhub | company news / financials / calendar | Skipped gracefully |

Every external call has timeout handling, exponential-backoff retries,
structured error logging, and an explicit degraded-mode status surfaced to the
signal. **No silent failures; bad data blocks signals rather than manufacturing
false confidence.**

---

## 6. API Specification (selected)

Base URL `http://<host>:8000` · interactive docs at `/docs`.

### 6.1 Endpoints
| Method | Path | Description |
|---|---|---|
| GET | `/health` | Provider availability, DB status, last-signal time |
| POST | `/analyze/{ticker}?mode=strict\|research` | Enqueue analysis job |
| GET | `/jobs/{job_id}` | Poll job state / result |
| GET | `/signals/latest` | Latest governed signal |
| GET | `/signals/history` | Signal ledger |
| GET | `/predictions/{signal_id}` | Full governance record |
| GET | `/risk/status` | Drift, rolling accuracy, kill switch |
| GET | `/paper-trades` | Paper-trade performance + trades |
| GET | `/model/status` | Registered model metrics |

### 6.2 Execution Model
Long-running analysis is asynchronous: `POST /analyze` returns a `job_id`;
clients poll `/jobs/{job_id}`. Work executes on Celery workers.

### 6.3 Signal Governance Record (per decision)
`signal_id` · ticker · timestamp · horizon · mode · decision · model outputs ·
news scores · base & adjusted risk scores · model agreement · data confidence ·
price/volume confirmation · expected edge (bps) · **gates passed/failed** ·
active threshold mode · position-sizing recommendation · kill-switch status ·
**data-quality status** · human-readable reason.

---

## 7. Risk Controls & Safety

| Control | Default |
|---|---|
| Max risk per trade | 0.5% of account |
| Max daily loss | 2.0% |
| Max weekly drawdown | 5.0% |
| Max consecutive losses | 2 (cooldown) |
| Min downside probability (SHORT, strict) | 0.65 |
| Min model agreement (strict) | 0.70 |
| Min data confidence (strict) | 0.75 |
| Price/volume confirmation | Required for SHORT |
| Earnings blackout | Configurable |
| **Live brokerage trading** | **Disabled — API refuses to start if enabled** |

Stop-loss is volatility-adjusted; position size scales with confidence ×
agreement × data confidence. The kill switch downgrades any SHORT to NO TRADE
when limits are breached and records a `risk_events` entry.

---

## 8. Validation Methodology

- **Walk-forward only** — expanding, time-ordered windows; never a random split.
- **Leakage controls** — backward-looking features (verified by an automated
  invariance test), label-only use of future data, timestamp-gated fundamentals,
  news treated as a live overlay (never backfilled).
- **Cost realism** — commission + slippage gating; expected edge must exceed
  round-trip cost.
- **Benchmarks** — random, naive-momentum, moving-average, logistic baselines.
- **Honest reporting** — representative single-ticker daily walk-forward AUC is
  ≈ 0.54. The platform is explicitly designed so the *risk-controlled decision*,
  not the raw predictor, is the value. Results are reported without inflation.

---

## 9. Performance & Operational Characteristics

| Metric | Characteristic |
|---|---|
| Cold analysis (first run / ticker) | ~10–15 s (ingest + train + score) |
| Warm analysis (cached model) | a few seconds |
| Historical capacity (daily) | ~6 years per ticker |
| Concurrency | Horizontal via additional Celery workers |
| Storage footprint | Low (SQLite dev); scales with PostgreSQL |
| Observability | JSON logs + audit tables + `/health`, `/risk/status` |

---

## 10. Deployment & Configuration

```bash
docker compose -f deploy/docker-compose.yml up --build          # api+worker+redis+dashboard
docker compose -f deploy/docker-compose.yml --profile postgres up --build
```
- SQLite → PostgreSQL by changing `DATABASE_URL` only.
- All thresholds, weights, providers, tickers, and horizons are config-driven
  (`settings · risk_limits · model_config · data_sources · thresholds`).
- Required vs optional environment variables validated at startup; no secrets in
  source.

---

## 11. Security & Compliance Posture

- Secrets via environment only; `.env` git-ignored; startup validation.
- Non-root container runtime.
- Full audit trail (predictions, decisions, risk events, system logs).
- **Compliance stance:** research/decision-support tool; **not** an order-routing
  or execution system; live trading disabled by default. Authentication and
  multi-tenant access control are roadmap items (see §13).

---

## 12. Limitations

- MVP target is **next-session daily** return, not a true intraday 12h model
  (requires a higher-frequency data provider to upgrade).
- Single-ticker focus per analysis; multi-ticker scale-out is roadmap.
- Core models do **not** train on news (insufficient leakage-free history);
  news is a live overlay until enough logged data accumulates for a meta-model.
- Free-tier data providers impose rate limits.
- All trading results are **hypothetical (paper)**.

---

## 13. Roadmap

| Horizon | Item |
|---|---|
| Near | True intraday horizon (Polygon/Alpaca); API authentication |
| Near | Learned ensemble weights via walk-forward |
| Mid | News meta-model trained from the live-logged dataset |
| Mid | Multi-ticker universe & portfolio-level risk |
| Mid | PostgreSQL production hardening; SHAP explanations in UI |
| Long | Supervised, extensively-validated real-capital research mode |

---

## 14. Disclaimer

DownsideIQ is provided for **research and educational purposes only**. It is
**not financial advice**, makes **no guarantee of profit**, and must not be used
to make real-money trading decisions without independent validation and
professional judgment. **Live trading is disabled by default and should remain
so until the system has undergone extensive validation.** Markets involve
substantial risk of loss.

---

<sub>DownsideIQ Product Specification Sheet · v1.0 · © 2026. "DownsideIQ" and the
DownsideIQ mark are used for project identification. All performance figures are
illustrative and derived from paper/backtest research.</sub>
