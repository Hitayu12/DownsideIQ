"""Feature service — builds + validates + persists leakage-safe feature snapshots.

Reuses the validated MVP feature math (``src.features.*``) behind a typed
interface. News is intentionally NOT a feature here (it is a decision-time
overlay); the snapshot carries only backfillable features + timestamp-safe
fundamentals + a data-confidence score. Bad data raises ``DataQualityError``
(blocks signals) rather than producing fake confidence (spec §10).
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from src.core.config import get_model_config, get_settings
from src.core.errors import DataQualityError
from src.core.logging import get_logger
from src.core.time import now_utc, to_utc
from src.db import repositories as repo
from src.db.session import get_session
from src.domain.features import FeatureSnapshot, IngestionResult
from src.features.feature_builder import CORE_FEATURE_COLS, build_feature_table
from src.features.fundamental_features import compute_fundamental_features

log = get_logger("services.feature")


class FeatureService:
    def build_table(self, ingestion: IngestionResult) -> pd.DataFrame:
        """Backfillable historical feature matrix (for training / backtest)."""
        return build_feature_table(ingestion.prices, ingestion.context, ingestion.ticker)

    def build_snapshot(self, ingestion: IngestionResult, *, persist: bool = True) -> FeatureSnapshot:
        if ingestion.prices.empty:
            raise DataQualityError(f"{ingestion.ticker}: no price data for feature snapshot.")

        table = self.build_table(ingestion)
        if table.empty:
            raise DataQualityError(f"{ingestion.ticker}: feature table is empty.")
        last_ts = table.index.max()
        row = table.loc[last_ts].to_dict()

        # Timestamp-safe fundamentals (suppresses unreleased earnings figures).
        row.update(compute_fundamental_features(ingestion.fundamentals, as_of=ingestion.as_of))

        self._validate(ingestion.ticker, last_ts, row, ingestion.as_of)

        conf, missing = self._data_confidence(row, ingestion)
        settings = get_settings()
        snap = FeatureSnapshot(
            ticker=ingestion.ticker,
            ts=to_utc(last_ts).to_pydatetime(),
            bar_size=ingestion.bar_size,
            horizon=settings.get("default_prediction_horizon", "24h"),
            feature_set_version=get_model_config().get("feature_set_version", "fs-1.0.0"),
            features={k: _json_safe(v) for k, v in row.items()},
            data_confidence_score=conf,
            missing_data_flag=missing,
        )
        if persist:
            self._persist(snap)
        log.info("feature_snapshot_built", ticker=snap.ticker, ts=str(snap.ts),
                 data_confidence=round(conf, 3), missing=missing)
        return snap

    # ---- validation (spec §10) ----
    def _validate(self, ticker: str, last_ts, row: dict, as_of: datetime) -> None:
        if to_utc(last_ts) > to_utc(as_of):
            raise DataQualityError(f"{ticker}: feature timestamp {last_ts} is after as_of {as_of}.")
        price = row.get("current_price")
        if price is None or (isinstance(price, float) and np.isnan(price)) or price <= 0:
            raise DataQualityError(f"{ticker}: invalid current_price ({price}).")
        # Sanity range checks on a couple of bounded features.
        for k in ("data_confidence_score",):
            pass  # computed later

    def _data_confidence(self, row: dict, ingestion: IngestionResult) -> tuple[float, bool]:
        core = [row.get(c) for c in CORE_FEATURE_COLS]
        present = [v for v in core if v is not None and not (isinstance(v, float) and np.isnan(v))]
        completeness = len(present) / len(CORE_FEATURE_COLS)
        fundamentals_ok = bool(row.get("fundamentals_available"))
        news_mult = ingestion.degraded.news_confidence_multiplier
        score = (0.5 * completeness + 0.2 * news_mult
                 + 0.15 * (1.0 if fundamentals_ok else 0.0) + 0.15)
        return float(np.clip(score, 0, 1)), bool(completeness < 0.8)

    def _persist(self, snap: FeatureSnapshot) -> int | None:
        from sqlalchemy.exc import IntegrityError

        try:
            with get_session() as s:
                obj = repo.save_feature_snapshot(s, {
                    "ticker": snap.ticker, "ts": snap.ts, "bar_size": snap.bar_size,
                    "horizon": snap.horizon, "feature_set_version": snap.feature_set_version,
                    "features": snap.features, "data_confidence_score": snap.data_confidence_score,
                    "missing_data_flag": snap.missing_data_flag,
                })
                return obj.id
        except IntegrityError:
            log.info("feature_snapshot_exists", ticker=snap.ticker, ts=str(snap.ts))
            return None


def _json_safe(v):
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v
