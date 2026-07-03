# DownsideIQ — Live Short-Horizon Equity Risk & Signal Engine

> **Prediction is not the edge. Risk-controlled decision-making is the edge.**

DownsideIQ is a **deployable, auditable research platform** that estimates 12–24h
**downside risk** for a stock by combining price action, volatility modeling,
market/sector context, timestamp-safe fundamentals, and a **live news event-risk
overlay** — and turns it into one of three governed decisions: `SHORT`, `WATCH`,
or `NO TRADE`. Every signal is timestamped, explained, persisted, and reconciled
against the realised outcome.

> ⚠️ **Research & risk-control platform. Not financial advice. No promise of
> profit. Live brokerage trading is disabled by default and must stay disabled.**

## Architecture (modular monolith)

```
Streamlit dashboard ──HTTP──▶ FastAPI backend ──enqueue──▶ Redis ──▶ Celery worker
   (API client)      ◀──JSON──  (/analyze, /signals…)   ◀──poll──        │
                                                                         ▼
ingestion → feature → model → news_scoring → ensemble → risk_engine → ledger
                                  (persisted to 11 SQLAlchemy tables)
```

Services (`src/services/`) have typed Pydantic interfaces and single
responsibilities. See **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**,
**[DATA_PIPELINE](docs/DATA_PIPELINE.md)**, **[MODELING](docs/MODELING.md)**,
**[RISK_ENGINE](docs/RISK_ENGINE.md)**, **[API](docs/API.md)**, and the
**[IMPLEMENTATION_PLAN](docs/IMPLEMENTATION_PLAN.md)**.

## Why this is not a generic stock predictor
Downside-risk framing (not price); **no look-ahead bias** (enforced + tested in
`tests/test_no_leakage.py`); **walk-forward** validation with costs; **no-trade
discipline**; full **accountability** (every signal logged before outcome, then
reconciled + attributed). Honest results: single-stock daily AUC ≈ 0.54 — the
value is the risk-controlled decision, not a magic predictor.

## Setup
```bash
python3 -m venv .venv_full && source .venv_full/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in keys (all optional; yfinance needs none)
python -m src.cli migrate     # create / migrate the database (Alembic)
```

### Environment (`.env`)
| Variable | Purpose | Required? |
|---|---|---|
| `TAVILY_API_KEY` | live news search | optional (price-only if absent) |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | LLM news scoring | optional (heuristic fallback) |
| `ALPHA_VANTAGE_API_KEY`, `FINNHUB_API_KEY` | fundamentals/earnings | optional |
| `DATABASE_URL` | SQLite now, Postgres later | default sqlite |
| `REDIS_URL` | Celery broker/backend | default localhost |
| `CELERY_EAGER` | run jobs inline w/o Redis (dev) | default false |
| `LIVE_TRADING_ENABLED` | **must stay false** | — |

Required vs optional keys are validated at startup (`validate_environment`).

## Run

**Docker (all services):**
```bash
docker compose -f deploy/docker-compose.yml up --build      # api+worker+redis+dashboard
docker compose -f deploy/docker-compose.yml --profile postgres up --build
```

**Locally:**
```bash
python -m src.cli run-api          # FastAPI  → http://localhost:8000  (/docs)
python -m src.cli run-worker       # Celery worker (needs Redis)
python -m src.cli run-dashboard    # Streamlit → http://localhost:8501
```

**CLI:**
```bash
python -m src.cli collect          --ticker NVDA
python -m src.cli train            --ticker NVDA
python -m src.cli analyze          --ticker NVDA --mode strict
python -m src.cli backtest         --ticker NVDA --mode research
python -m src.cli update-outcomes
pytest                              # test suite
```

## Risk controls
Signal-quality gate (mode-aware), volatility-aware position sizing,
stop-loss/take-profit, daily-loss & weekly-drawdown limits, consecutive-loss
cooldown, **kill switch** (downgrades SHORT→NO TRADE), full §12 governance on
every signal. Strict (live) vs research (exploratory) modes are never mixed.

## Data-leakage rules
Backward-looking features only; the label is the only future data, built after
the prediction is logged; fundamentals suppressed until publicly released; news
is a live overlay (never backfilled). Bad data **blocks** signals
(`DataQualityError`) rather than producing fake confidence.

## Limitations
MVP target is next-session daily (not true intraday 12h — needs a stronger
intraday provider); single-ticker focus; historical models don't train on news
(insufficient leakage-free history yet); free-tier provider rate limits; paper
results are hypothetical.

## Roadmap
Intraday horizon (Polygon/Alpaca) · news meta-model from logged data · learned
ensemble weights · API auth · Postgres in prod · multi-ticker scale-out · SHAP.

## Disclaimer
For research/education only. **Not financial advice.** No profit guarantee.
Live trading disabled by default and must remain so until extensive validation.
Markets carry substantial risk of loss.
