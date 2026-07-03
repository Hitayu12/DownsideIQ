"""Repository layer — typed data-access objects over the ORM.

Services depend on these functions, never on raw SQL or session internals,
keeping persistence swappable and testable. All writes are idempotent where a
natural key exists (price bars, feature snapshots).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from src.core.time import now_utc, to_utc
from src.db import models as m


# --------------------------- prices --------------------------- #
def upsert_price_bars(session: Session, ticker: str, bar_size: str, rows: list[dict]) -> int:
    """Idempotent insert of OHLCV bars (ignore duplicates on natural key)."""
    if not rows:
        return 0
    payload = [{"ticker": ticker.upper(), "bar_size": bar_size, **r} for r in rows]
    stmt = sqlite_insert(m.RawPriceData).values(payload)
    stmt = stmt.on_conflict_do_nothing(index_elements=["ticker", "bar_size", "ts"])
    session.execute(stmt)
    return len(payload)


def latest_price_ts(session: Session, ticker: str, bar_size: str) -> datetime | None:
    return session.scalar(
        select(m.RawPriceData.ts).where(
            m.RawPriceData.ticker == ticker.upper(), m.RawPriceData.bar_size == bar_size
        ).order_by(m.RawPriceData.ts.desc()).limit(1)
    )


# --------------------------- news --------------------------- #
def save_raw_news(session: Session, items: list[dict]) -> list[m.RawNewsResult]:
    objs = [m.RawNewsResult(**it) for it in items]
    session.add_all(objs)
    session.flush()
    return objs


def save_news_scores(session: Session, scores: list[dict]) -> int:
    session.add_all([m.StructuredNewsScore(**s) for s in scores])
    return len(scores)


# --------------------------- features --------------------------- #
def save_feature_snapshot(session: Session, snap: dict) -> m.FeatureSnapshot:
    obj = m.FeatureSnapshot(**snap)
    session.add(obj)
    session.flush()
    return obj


# --------------------------- predictions --------------------------- #
def save_predictions(session: Session, preds: list[dict]) -> int:
    session.add_all([m.ModelPrediction(**p) for p in preds])
    return len(preds)


# --------------------------- signals --------------------------- #
def save_signal(session: Session, signal: dict) -> m.FinalSignal:
    obj = m.FinalSignal(**signal)
    session.add(obj)
    session.flush()
    return obj


def latest_signal(session: Session, ticker: str | None = None, mode: str = "strict") -> m.FinalSignal | None:
    q = select(m.FinalSignal).where(m.FinalSignal.mode == mode)
    if ticker:
        q = q.where(m.FinalSignal.ticker == ticker.upper())
    return session.scalar(q.order_by(m.FinalSignal.ts.desc(), m.FinalSignal.created_at.desc()).limit(1))


def signal_history(session: Session, ticker: str | None = None, mode: str = "strict",
                   limit: int = 200) -> Sequence[m.FinalSignal]:
    q = select(m.FinalSignal).where(m.FinalSignal.mode == mode)
    if ticker:
        q = q.where(m.FinalSignal.ticker == ticker.upper())
    return list(session.scalars(q.order_by(m.FinalSignal.ts.desc()).limit(limit)))


def get_signal(session: Session, signal_id: str) -> m.FinalSignal | None:
    return session.get(m.FinalSignal, signal_id)


def predictions_for_signal(session: Session, signal_id: str) -> Sequence[m.ModelPrediction]:
    return list(session.scalars(select(m.ModelPrediction).where(m.ModelPrediction.signal_id == signal_id)))


def update_signal_outcome(session: Session, signal_id: str, actual_return: float, correct: bool) -> None:
    sig = session.get(m.FinalSignal, signal_id)
    if sig:
        sig.actual_return_24h = actual_return
        sig.prediction_correct = correct
        sig.updated_at = now_utc()


def open_signals_without_outcome(session: Session, mode: str | None = None) -> Sequence[m.FinalSignal]:
    q = select(m.FinalSignal).where(m.FinalSignal.actual_return_24h.is_(None))
    if mode:
        q = q.where(m.FinalSignal.mode == mode)
    return list(session.scalars(q))


# --------------------------- paper trades --------------------------- #
def save_paper_trade(session: Session, trade: dict) -> m.PaperTrade:
    obj = m.PaperTrade(**trade)
    session.merge(obj)
    return obj


def paper_trades(session: Session, ticker: str | None = None, mode: str | None = None,
                 limit: int = 500) -> Sequence[m.PaperTrade]:
    q = select(m.PaperTrade)
    if ticker:
        q = q.where(m.PaperTrade.ticker == ticker.upper())
    if mode:
        q = q.where(m.PaperTrade.mode == mode)
    return list(session.scalars(q.order_by(m.PaperTrade.entry_time.desc()).limit(limit)))


# --------------------------- attribution / risk / perf / logs --------------------------- #
def save_attribution(session: Session, attr: dict) -> None:
    session.add(m.PostTradeAttribution(**attr))


def log_risk_event(session: Session, event_type: str, *, ticker: str | None = None,
                   severity: str = "info", detail: dict | None = None) -> None:
    session.add(m.RiskEvent(ts=now_utc(), ticker=ticker, event_type=event_type,
                            severity=severity, detail=detail))


def record_model_performance(session: Session, rows: list[dict]) -> int:
    session.add_all([m.ModelPerformance(**r) for r in rows])
    return len(rows)


def write_system_log(session: Session, *, level: str, logger: str, event: str,
                     ticker: str | None = None, payload: dict | None = None) -> None:
    session.add(m.SystemLog(ts=now_utc(), level=level, logger=logger, event=event,
                            ticker=ticker, payload=payload))
