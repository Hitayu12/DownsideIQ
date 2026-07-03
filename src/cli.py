"""DownsideIQ CLI (spec §14). Typer-based.

Commands:
    collect          ingest + snapshot for a ticker
    train            train + version models for a ticker
    analyze          run the full pipeline -> signal (strict|research)
    update-outcomes  reconcile matured signals with realised returns
    backtest         out-of-sample paper backtest (research/strict)
    run-api          start the FastAPI backend (uvicorn)
    run-worker       start the Celery worker
    run-dashboard    start the Streamlit dashboard
    migrate          apply Alembic migrations
"""
from __future__ import annotations

import subprocess
import sys

import typer

from src.core.logging import get_logger

app = typer.Typer(add_completion=False, help="DownsideIQ — downside-risk research platform.")
log = get_logger("cli")


@app.command()
def collect(ticker: str = typer.Option(..., "--ticker", "-t")):
    from src.db.session import init_db
    from src.services.feature_service import FeatureService
    from src.services.ingestion_service import IngestionService

    init_db()
    ing = IngestionService().ingest(ticker)
    snap = FeatureService().build_snapshot(ing)
    typer.echo(f"collected {ing.ticker}: {len(ing.prices)} bars, snapshot {snap.ts} "
               f"(data_confidence={snap.data_confidence_score:.2f}, quality={ing.degraded.status.value})")


@app.command()
def train(ticker: str = typer.Option(..., "--ticker", "-t")):
    from src.db.session import init_db
    from src.services.ingestion_service import IngestionService
    from src.services.model_service import ModelService

    init_db()
    ing = IngestionService().ingest(ticker)
    metrics = ModelService().train(ing)
    typer.echo(f"trained {ticker}: walk-forward AUC={metrics.get('auc_mean', float('nan')):.3f} "
               f"Brier={metrics.get('brier_mean', float('nan')):.3f}")


@app.command()
def analyze(ticker: str = typer.Option(..., "--ticker", "-t"),
            mode: str = typer.Option("strict", "--mode", "-m")):
    from src.db.session import init_db
    from src.pipeline.orchestrator import analyze as run_analyze

    init_db()
    d = run_analyze(ticker, mode=mode)
    typer.echo(f"\n{d['ticker']} [{mode}] -> {d['decision']}  (signal {d['signal_id']})")
    typer.echo(f"  adj P(downside)={d['adjusted_p_downside']:.3f}  risk={d['adjusted_downside_risk_score']:.3f}"
               f"  agreement={d['model_agreement_score']:.3f}  data_quality={d['data_quality']}")
    typer.echo(f"  reason: {d['governance']['reason']}")


@app.command("update-outcomes")
def update_outcomes():
    from src.db.session import init_db
    from src.tasks.jobs import update_outcomes_task

    init_db()
    res = update_outcomes_task.run()      # synchronous
    typer.echo(f"outcomes updated: {res['updated']}")


@app.command()
def backtest(ticker: str = typer.Option(..., "--ticker", "-t"),
             mode: str = typer.Option("research", "--mode", "-m")):
    from src.db.session import init_db
    from src.services.ingestion_service import IngestionService
    from src.trading.paper_backtest import EXPLORATORY_OVERRIDES, run_paper_backtest

    init_db()
    ing = IngestionService().ingest(ticker, persist=False)
    collected = {"prices": ing.prices, "context": ing.context}
    overrides = EXPLORATORY_OVERRIDES if mode == "research" else None
    summary = run_paper_backtest(ticker, collected, mode=mode, gate_overrides=overrides,
                                 trade_decisions=("SHORT",) if mode == "research" else ())
    _persist_backtest_trades(ticker, mode)
    m = summary.get("trading_metrics", {})
    typer.echo(f"backtest {ticker} [{mode}]: decisions={summary['decision_counts']} "
               f"trades={summary.get('n_trades', 0)} hit_rate={m.get('hit_rate', float('nan')):.2f} "
               f"PF={m.get('profit_factor', float('nan')):.2f}")


def _persist_backtest_trades(ticker: str, mode: str) -> None:
    """Load the backtest's paper-trade output into the DB for the API/dashboard."""
    import pandas as pd

    from src.core.config import data_dir
    from src.services.paper_trading_service import PaperTradingService

    path = data_dir() / "predictions" / "paper_trades.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    svc = PaperTradingService()
    for _, r in df.iterrows():
        svc._persist({**r.to_dict(), "mode": mode})


@app.command("run-api")
def run_api(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run("src.api.app:app", host=host, port=port, reload=False)


@app.command("run-worker")
def run_worker():
    subprocess.run([sys.executable, "-m", "celery", "-A", "src.tasks.celery_app",
                    "worker", "--loglevel=info"], check=False)


@app.command("run-dashboard")
def run_dashboard(port: int = 8501):
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/dashboard/app.py",
                    "--server.port", str(port)], check=False)


@app.command()
def migrate():
    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], check=False)


if __name__ == "__main__":
    app()
