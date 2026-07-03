"""FastAPI backend (spec §2) — all business logic lives in services; this layer
only validates input, enqueues async work, and serializes responses.

Endpoints:
  GET  /health
  POST /analyze/{ticker}           -> enqueue Celery job
  GET  /jobs/{job_id}              -> poll job result
  GET  /signals/latest
  GET  /signals/history
  GET  /predictions/{signal_id}
  GET  /risk/status
  GET  /paper-trades
  GET  /model/status
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.schemas import HealthResponse, JobResponse, JobResult
from src.core.config import get_thresholds, project_root, validate_environment
from src.core.logging import get_logger
from src.services.ledger_service import LedgerService
from src.services.monitoring_service import MonitoringService
from src.services.paper_trading_service import PaperTradingService

log = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    status = validate_environment()          # fatal on misconfig
    from src.db.session import init_db
    init_db()
    log.info("api_startup", **{k: v for k, v in status.items() if k != "optional_present"})
    yield


app = FastAPI(title="DownsideIQ API", version="1.0.0",
              description="Short-horizon equity downside-risk research platform.",
              lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_VALID_MODES = ("strict", "research")
_WEB_DIR = project_root() / "web"
if _WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_WEB_DIR)), name="static")


@app.get("/", include_in_schema=False)
def index():
    """Serve the single-page web terminal."""
    idx = _WEB_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    raise HTTPException(404, "web UI not built")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    raise HTTPException(404)


@app.get("/health", response_model=HealthResponse)
def health():
    return MonitoringService().health()


@app.post("/analyze/{ticker}", response_model=JobResponse)
def analyze(ticker: str, mode: str = Query("strict")):
    if mode not in _VALID_MODES:
        raise HTTPException(400, f"mode must be one of {_VALID_MODES}")
    from src.tasks.jobs import analyze_task

    job = analyze_task.delay(ticker.upper(), mode)
    return JobResponse(job_id=job.id, status="queued", mode=mode, ticker=ticker.upper())


@app.get("/jobs/{job_id}", response_model=JobResult)
def job_status(job_id: str):
    from celery.result import AsyncResult
    from src.tasks.celery_app import celery_app

    res = AsyncResult(job_id, app=celery_app)
    payload = res.result if res.ready() and res.successful() else None
    if res.ready() and not res.successful():
        payload = {"error": str(res.result)[:300]}
    return JobResult(job_id=job_id, state=res.state, ready=res.ready(), result=payload)


@app.get("/signals/latest")
def signals_latest(ticker: str | None = None, mode: str = "strict"):
    sig = LedgerService().latest(ticker=ticker, mode=mode)
    if not sig:
        raise HTTPException(404, "no signals yet")
    return sig


@app.get("/signals/history")
def signals_history(ticker: str | None = None, mode: str = "strict", limit: int = 200):
    return {"signals": LedgerService().history(ticker=ticker, mode=mode, limit=limit)}


@app.get("/predictions/{signal_id}")
def predictions(signal_id: str):
    sig = LedgerService().get(signal_id)
    if not sig:
        raise HTTPException(404, "signal not found")
    return sig


@app.get("/risk/status")
def risk_status(ticker: str | None = None, mode: str = "strict"):
    return MonitoringService().drift(ticker=ticker, mode=mode)


@app.get("/paper-trades")
def paper_trades(ticker: str | None = None, mode: str = "research"):
    svc = PaperTradingService()
    return {"performance": svc.performance(ticker=ticker, mode=mode),
            "trades": svc.list_trades(ticker=ticker, mode=mode)}


@app.get("/model/status")
def model_status(ticker: str | None = None):
    return {"performance": MonitoringService().model_status(ticker=ticker),
            "active_mode": get_thresholds().get("default_mode")}
