"""XGBoost downside classifier (Council 2, Model 1 — spec §10.1).

Answers: "What is the probability of a meaningful downside move over the next
12-24h (next session for the daily MVP)?" XGBoost handles NaN natively, so
warmup gaps in features don't need imputation. Class imbalance is handled via
``scale_pos_weight``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

log = get_logger("models.downside")

_DEFAULT_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=0,
)


class DownsideClassifier:
    """Thin wrapper around xgboost.XGBClassifier with persistence + SHAP-lite."""

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = {**_DEFAULT_PARAMS, **(params or {})}
        self.model = None
        self.feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DownsideClassifier":
        from xgboost import XGBClassifier

        self.feature_names = list(X.columns)
        pos = float((y == 1).sum())
        neg = float((y == 0).sum())
        spw = (neg / pos) if pos > 0 else 1.0
        self.model = XGBClassifier(scale_pos_weight=spw, **self.params)
        self.model.fit(X.values, y.values)
        log.info("Trained downside classifier on %d samples (pos=%d, neg=%d, spw=%.2f).",
                 len(y), int(pos), int(neg), spw)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained/loaded.")
        X = X[self.feature_names] if self.feature_names else X
        return self.model.predict_proba(X.values)[:, 1]

    def predict_one(self, row: dict[str, Any]) -> float:
        """Predict p_downside for a single feature row (dict)."""
        X = pd.DataFrame([{k: row.get(k, np.nan) for k in self.feature_names}])
        return float(self.predict_proba(X)[0])

    def top_features(self, k: int = 10) -> list[tuple[str, float]]:
        if self.model is None:
            return []
        imp = self.model.feature_importances_
        pairs = sorted(zip(self.feature_names, imp), key=lambda t: t[1], reverse=True)
        return [(n, float(v)) for n, v in pairs[:k]]

    # --- Persistence ---
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path.with_suffix(".json")))
        with path.with_suffix(".meta.json").open("w", encoding="utf-8") as fh:
            json.dump({"feature_names": self.feature_names, "params": self.params}, fh, indent=2)
        log.info("Saved downside classifier -> %s", path.with_suffix(".json"))
        return path.with_suffix(".json")

    @classmethod
    def load(cls, path: str | Path) -> "DownsideClassifier":
        from xgboost import XGBClassifier

        path = Path(path)
        meta_path = path.with_suffix(".meta.json")
        with meta_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
        obj = cls(params=meta.get("params"))
        obj.feature_names = meta.get("feature_names", [])
        obj.model = XGBClassifier()
        obj.model.load_model(str(path.with_suffix(".json")))
        return obj
