"""Model service (spec §1, §11) — train + versioned inference for Council 2.

Wraps the validated model classes (downside classifier, GARCH, quantile) behind
a typed interface, attaches full versioning metadata to every prediction, and
persists predictions for audit. Training records walk-forward metrics to the
model registry / ``model_performance``.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from src.backtesting.metrics import aggregate_fold_metrics, classification_metrics
from src.backtesting.walk_forward import walk_forward_splits
from src.core.config import get_model_config
from src.core.logging import get_logger
from src.core.time import now_utc, to_utc
from src.db import repositories as repo
from src.db.session import get_session
from src.domain.features import IngestionResult
from src.domain.predictions import CouncilOutputs, ModelPrediction
from src.features.feature_builder import CORE_FEATURE_COLS, build_feature_table
from src.models.downside_classifier import DownsideClassifier
from src.models.quantile_model import QuantileDownsideModel
from src.models.registry import ModelRegistry
from src.models.target_builder import build_targets
from src.models.volatility_model import fit_and_forecast

log = get_logger("services.model")


class ModelService:
    def __init__(self, registry: ModelRegistry | None = None):
        self.registry = registry or ModelRegistry()
        self.cfg = get_model_config()
        self.fsv = self.cfg.get("feature_set_version", "fs-1.0.0")

    # ---------------- data prep ---------------- #
    def _prepare(self, ingestion: IngestionResult):
        table = build_feature_table(ingestion.prices, ingestion.context, ingestion.ticker)
        targets = build_targets(ingestion.prices)
        cols = [c for c in CORE_FEATURE_COLS if c in table.columns]
        X = table[cols]
        y_label = targets["downside_label"].reindex(X.index)
        y_future = targets["future_return"].reindex(X.index)
        keep = y_label.notna() & y_future.notna() & (X.notna().mean(axis=1) >= 0.5)
        return X.loc[keep], y_label.loc[keep].astype(int), y_future.loc[keep]

    # ---------------- training ---------------- #
    def train(self, ingestion: IngestionResult, mode: str = "strict") -> dict[str, Any]:
        ticker = ingestion.ticker
        X, y_label, y_future = self._prepare(ingestion)
        td = now_utc()

        # Walk-forward evaluation (real OOS metrics, never from mocks).
        folds = []
        for sp in walk_forward_splits(len(X), **{k: self.cfg["walk_forward"][k]
                                                 for k in ("n_splits", "min_train") if k in self.cfg.get("walk_forward", {})}):
            clf = DownsideClassifier(self.cfg["downside_classifier"]["params"]).fit(
                X.iloc[sp.train_idx], y_label.iloc[sp.train_idx])
            p = clf.predict_proba(X.iloc[sp.test_idx])
            folds.append(classification_metrics(y_label.iloc[sp.test_idx].values, p))
        metrics = aggregate_fold_metrics(folds)

        # Final fit on all available history + persist artifacts.
        clf = DownsideClassifier(self.cfg["downside_classifier"]["params"]).fit(X, y_label)
        qm = QuantileDownsideModel(tuple(self.cfg["quantile_model"]["quantiles"]),
                                   self.cfg["quantile_model"]["params"].get("max_iter", 300)).fit(X, y_future)
        clf.save(self.registry.artifact_path(ticker, "downside_xgb"))
        qm.save(self.registry.artifact_path(ticker, "quantile_hgb"))

        self.registry.register(ticker, "downside_xgb", self.cfg["downside_classifier"]["version"],
                               self.fsv, td, metrics, mode)
        self.registry.register(ticker, "quantile_hgb", self.cfg["quantile_model"]["version"],
                               self.fsv, td, None, mode)
        log.info("models_trained", ticker=ticker, n=len(X),
                 auc=round(metrics.get("auc_mean", float("nan")), 3))
        return metrics

    def load_or_train(self, ingestion: IngestionResult):
        ticker = ingestion.ticker
        clf_path = self.registry.artifact_path(ticker, "downside_xgb").with_suffix(".json")
        qm_path = self.registry.artifact_path(ticker, "quantile_hgb").with_suffix(".joblib")
        if clf_path.exists() and qm_path.exists():
            try:
                return (DownsideClassifier.load(self.registry.artifact_path(ticker, "downside_xgb")),
                        QuantileDownsideModel.load(self.registry.artifact_path(ticker, "quantile_hgb")))
            except Exception as exc:  # noqa: BLE001
                log.warning("model_load_failed_retraining", error=str(exc)[:160])
        self.train(ingestion)
        return (DownsideClassifier.load(self.registry.artifact_path(ticker, "downside_xgb")),
                QuantileDownsideModel.load(self.registry.artifact_path(ticker, "quantile_hgb")))

    # ---------------- inference ---------------- #
    def predict(self, ingestion: IngestionResult, snapshot, *, persist: bool = True) -> CouncilOutputs:
        ticker = ingestion.ticker
        ts = to_utc(snapshot.ts).to_pydatetime()
        clf, qm = self.load_or_train(ingestion)
        feats = snapshot.features

        p = float(clf.predict_one(feats))
        close = ingestion.prices.sort_index()["close"].astype(float)
        log_ret = np.log(close / close.shift(1))
        garch = fit_and_forecast(log_ret, vol=self.cfg["volatility_model"]["vol"], alpha=0.05)
        tail = qm.predict_one(feats)

        clf_md = self.registry.metadata(ticker, "downside_xgb")
        qm_md = self.registry.metadata(ticker, "quantile_hgb")
        vol_md = self.registry.metadata(ticker, "vol_garch")

        def _pred(md, output) -> ModelPrediction:
            return ModelPrediction(
                model_name=md["model_name"], model_version=md["model_version"],
                training_date=to_utc(md["training_date"]).to_pydatetime() if md.get("training_date") else None,
                feature_set_version=md.get("feature_set_version", self.fsv),
                prediction_timestamp=ts, ticker=ticker, output=output,
            )

        outputs = CouncilOutputs(
            ticker=ticker, timestamp=ts,
            classifier=_pred(clf_md, {"p_downside": p, "xgb_confidence": abs(p - 0.5) * 2,
                                      "top_features": clf.top_features(8)}),
            volatility=_pred(vol_md, garch),
            quantile=_pred(qm_md, tail),
        )
        if persist:
            self._persist(outputs)
        log.info("council_predicted", ticker=ticker, p_downside=round(p, 4),
                 garch_dvr=round(garch.get("downside_volatility_risk", 0), 3),
                 tail=round(tail.get("downside_tail_score", 0), 3))
        return outputs

    def _persist(self, outputs: CouncilOutputs) -> None:
        rows = []
        for mp in (outputs.classifier, outputs.volatility, outputs.quantile):
            rows.append({
                "ticker": mp.ticker, "ts": mp.prediction_timestamp,
                "model_name": mp.model_name, "model_version": mp.model_version,
                "training_date": mp.training_date, "feature_set_version": mp.feature_set_version,
                "prediction_timestamp": mp.prediction_timestamp, "output": mp.output,
            })
        with get_session() as s:
            repo.save_predictions(s, rows)
