"""SQLAlchemy 2.0 ORM models — the 11 audit tables (spec §3).

Design notes:
- UTC, tz-aware ``DateTime(timezone=True)`` everywhere.
- ``mode`` column on signals/trades/performance separates strict vs research.
- Lineage FKs: signal -> feature_snapshot, prediction -> feature_snapshot,
  trade/attribution -> signal — full auditability.
- ``JSON`` columns work on both SQLite (MVP) and PostgreSQL (prod) unchanged.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _now_col() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now())


class RawPriceData(Base):
    __tablename__ = "raw_price_data"
    __table_args__ = (UniqueConstraint("ticker", "bar_size", "ts", name="uq_price_bar"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    bar_size: Mapped[str] = mapped_column(String(8))
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    adj_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="yfinance")
    created_at: Mapped[datetime] = _now_col()


class RawNewsResult(Base):
    __tablename__ = "raw_news_results"
    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    scope: Mapped[str] = mapped_column(String(16))            # company | macro
    query: Mapped[str | None] = mapped_column(Text, nullable=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    provider: Mapped[str] = mapped_column(String(32))
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = _now_col()


class StructuredNewsScore(Base):
    __tablename__ = "structured_news_scores"
    id: Mapped[int] = mapped_column(primary_key=True)
    raw_news_id: Mapped[int | None] = mapped_column(ForeignKey("raw_news_results.id"), nullable=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    scope: Mapped[str] = mapped_column(String(16))
    event_type: Mapped[str] = mapped_column(String(48))
    sentiment_score: Mapped[float] = mapped_column(Float)
    relevance_score: Mapped[float] = mapped_column(Float)
    credibility_score: Mapped[float] = mapped_column(Float)
    recency_score: Mapped[float] = mapped_column(Float)
    company_specificity_score: Mapped[float] = mapped_column(Float)
    expected_direction: Mapped[str] = mapped_column(String(12))
    expected_impact_score: Mapped[float] = mapped_column(Float)
    confidence_score: Mapped[float] = mapped_column(Float)
    scorer: Mapped[str] = mapped_column(String(16))           # gemini | heuristic
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    created_at: Mapped[datetime] = _now_col()


class FeatureSnapshot(Base):
    __tablename__ = "feature_snapshots"
    __table_args__ = (UniqueConstraint("ticker", "ts", "bar_size", name="uq_feature_snapshot"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    bar_size: Mapped[str] = mapped_column(String(8))
    horizon: Mapped[str] = mapped_column(String(8))
    feature_set_version: Mapped[str] = mapped_column(String(24))
    features: Mapped[dict] = mapped_column(JSON)
    data_confidence_score: Mapped[float] = mapped_column(Float)
    missing_data_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = _now_col()
    predictions: Mapped[list["ModelPrediction"]] = relationship(back_populates="snapshot")


class ModelPrediction(Base):
    __tablename__ = "model_predictions"
    id: Mapped[int] = mapped_column(primary_key=True)
    feature_snapshot_id: Mapped[int | None] = mapped_column(ForeignKey("feature_snapshots.id"), nullable=True)
    signal_id: Mapped[str | None] = mapped_column(ForeignKey("final_signals.signal_id"), nullable=True, index=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    model_name: Mapped[str] = mapped_column(String(48))
    model_version: Mapped[str] = mapped_column(String(24))
    training_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    feature_set_version: Mapped[str] = mapped_column(String(24))
    prediction_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    output: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = _now_col()
    snapshot: Mapped["FeatureSnapshot"] = relationship(back_populates="predictions")


class FinalSignal(Base):
    __tablename__ = "final_signals"
    signal_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    horizon: Mapped[str] = mapped_column(String(8))
    mode: Mapped[str] = mapped_column(String(12), index=True)   # strict | research
    decision: Mapped[str] = mapped_column(String(12), index=True)
    p_downside: Mapped[float | None] = mapped_column(Float, nullable=True)
    adjusted_p_downside: Mapped[float | None] = mapped_column(Float, nullable=True)
    base_downside_risk_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    adjusted_downside_risk_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_agreement_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    data_confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_volume_confirmation: Mapped[float | None] = mapped_column(Float, nullable=True)
    news_shift_logodds: Mapped[float | None] = mapped_column(Float, nullable=True)
    expected_edge_bps: Mapped[float | None] = mapped_column(Float, nullable=True)
    uncertainty_penalty: Mapped[float | None] = mapped_column(Float, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    governance: Mapped[dict] = mapped_column(JSON)             # full §12 record
    data_quality: Mapped[str] = mapped_column(String(12))     # ok | degraded | blocked
    kill_switch_active: Mapped[bool] = mapped_column(Boolean, default=False)
    feature_snapshot_id: Mapped[int | None] = mapped_column(ForeignKey("feature_snapshots.id"), nullable=True)
    # Outcome (filled by update-outcomes):
    actual_return_24h: Mapped[float | None] = mapped_column(Float, nullable=True)
    prediction_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime] = _now_col()
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class PaperTrade(Base):
    __tablename__ = "paper_trades"
    trade_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    signal_id: Mapped[str | None] = mapped_column(ForeignKey("final_signals.signal_id"), nullable=True, index=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    mode: Mapped[str] = mapped_column(String(12), index=True)
    side: Mapped[str] = mapped_column(String(8), default="short")
    entry_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    entry_price: Mapped[float] = mapped_column(Float)
    position_size: Mapped[float] = mapped_column(Float)
    position_notional: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_reason: Mapped[str | None] = mapped_column(String(24), nullable=True)
    return_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    result: Mapped[str | None] = mapped_column(String(8), nullable=True)
    market_regime: Mapped[str | None] = mapped_column(String(16), nullable=True)
    attribution: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = _now_col()
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class PostTradeAttribution(Base):
    __tablename__ = "post_trade_attribution"
    id: Mapped[int] = mapped_column(primary_key=True)
    signal_id: Mapped[str] = mapped_column(ForeignKey("final_signals.signal_id"), index=True)
    ticker: Mapped[str] = mapped_column(String(16))
    actual_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    market_component: Mapped[float | None] = mapped_column(Float, nullable=True)
    sector_component: Mapped[float | None] = mapped_column(Float, nullable=True)
    company_component: Mapped[float | None] = mapped_column(Float, nullable=True)
    dominant_driver: Mapped[str | None] = mapped_column(String(24), nullable=True)
    prediction_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    lesson: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = _now_col()


class RiskEvent(Base):
    __tablename__ = "risk_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    ticker: Mapped[str | None] = mapped_column(String(16), nullable=True)
    event_type: Mapped[str] = mapped_column(String(32))       # kill_switch | limit_breach | gate_reject
    severity: Mapped[str] = mapped_column(String(12), default="info")
    detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = _now_col()


class ModelPerformance(Base):
    __tablename__ = "model_performance"
    id: Mapped[int] = mapped_column(primary_key=True)
    model_name: Mapped[str] = mapped_column(String(48), index=True)
    model_version: Mapped[str] = mapped_column(String(24))
    training_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    feature_set_version: Mapped[str] = mapped_column(String(24))
    metric_name: Mapped[str] = mapped_column(String(48))
    metric_value: Mapped[float] = mapped_column(Float)
    window: Mapped[str | None] = mapped_column(String(24), nullable=True)
    mode: Mapped[str] = mapped_column(String(12), default="strict")
    created_at: Mapped[datetime] = _now_col()


class SystemLog(Base):
    __tablename__ = "system_logs"
    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    level: Mapped[str] = mapped_column(String(12))
    logger: Mapped[str] = mapped_column(String(64))
    event: Mapped[str] = mapped_column(String(128))
    ticker: Mapped[str | None] = mapped_column(String(16), nullable=True)
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = _now_col()
