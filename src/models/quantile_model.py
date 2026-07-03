"""Quantile-regression downside-tail model (Council 2, Model 3 — spec §10.3).

Answers: "If the stock drops, how bad could the downside be?" Predicts low
quantiles (5th / 10th percentile) of the next-session return. Uses
``HistGradientBoostingRegressor`` with the pinball (quantile) loss because it
handles NaN features natively (matching the classifier) and captures
nonlinearities. Output feeds the downside-tail term of the ensemble.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

log = get_logger("models.quantile")


class QuantileDownsideModel:
    """Two quantile regressors (5% and 10%) over the feature matrix."""

    def __init__(self, quantiles: tuple[float, float] = (0.05, 0.10), max_iter: int = 300):
        self.quantiles = quantiles
        self.max_iter = max_iter
        self.models: dict[float, object] = {}
        self.feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, future_return: pd.Series) -> "QuantileDownsideModel":
        from sklearn.ensemble import HistGradientBoostingRegressor

        self.feature_names = list(X.columns)
        for q in self.quantiles:
            m = HistGradientBoostingRegressor(
                loss="quantile", quantile=q, max_iter=self.max_iter,
                learning_rate=0.03, max_depth=3, l2_regularization=1.0,
            )
            m.fit(X.values, future_return.values)
            self.models[q] = m
        log.info("Trained quantile model (q=%s) on %d samples.", self.quantiles, len(X))
        return self

    def predict_one(self, row: dict) -> dict[str, float]:
        X = pd.DataFrame([{k: row.get(k, np.nan) for k in self.feature_names}])
        preds = {q: float(self.models[q].predict(X.values)[0]) for q in self.quantiles}
        q5 = preds.get(0.05, np.nan)
        q10 = preds.get(0.10, np.nan)
        # downside_tail_score: magnitude of the 5% quantile loss, squashed to 0..1.
        tail = float(np.clip(-q5 / 0.05, 0, 1)) if q5 == q5 else np.nan
        return {
            "predicted_5pct_return": q5,
            "predicted_10pct_return": q10,
            "downside_tail_score": tail,
            "expected_worst_case_range": [q5, q10],
        }

    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": self.models, "feature_names": self.feature_names,
                     "quantiles": self.quantiles}, path.with_suffix(".joblib"))
        with path.with_suffix(".meta.json").open("w", encoding="utf-8") as fh:
            json.dump({"feature_names": self.feature_names, "quantiles": list(self.quantiles)}, fh)
        return path.with_suffix(".joblib")

    @classmethod
    def load(cls, path: str | Path) -> "QuantileDownsideModel":
        import joblib

        path = Path(path)
        blob = joblib.load(path.with_suffix(".joblib"))
        obj = cls(quantiles=tuple(blob["quantiles"]))
        obj.models = blob["models"]
        obj.feature_names = blob["feature_names"]
        return obj
