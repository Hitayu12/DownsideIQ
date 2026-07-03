"""Training + walk-forward evaluation for the downside classifier (Phase 4).

Assembles (features, target) with the correct time contract, evaluates via
walk-forward, trains a final model on all available history, and produces
``p_downside`` for the latest bar. Pure offline (uses saved/collected data).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.backtesting.metrics import aggregate_fold_metrics, classification_metrics
from src.backtesting.walk_forward import walk_forward_splits
from src.features.feature_builder import CORE_FEATURE_COLS, build_feature_table, build_live_feature_row
from src.models.downside_classifier import DownsideClassifier
from src.models.quantile_model import QuantileDownsideModel
from src.models.target_builder import build_targets
from src.models.volatility_model import fit_and_forecast
from src.utils.config_loader import project_root
from src.utils.logging_utils import get_logger

log = get_logger("models.training")

_MIN_FEATURE_COVERAGE = 0.5   # drop rows where >50% of features are NaN (warmup)


def model_path(ticker: str):
    return project_root() / "models_store" / f"{ticker.upper()}_downside"


def prepare_xy(collected: dict[str, Any], ticker: str) -> tuple[pd.DataFrame, pd.Series]:
    """Build aligned (X, y) with backward-looking features and forward label."""
    prices = collected["prices"]
    table = build_feature_table(prices, collected.get("context", {}), ticker)
    targets = build_targets(prices)

    cols = [c for c in CORE_FEATURE_COLS if c in table.columns]
    X = table[cols].copy()
    y = targets["downside_label"].reindex(X.index)

    # Drop rows with undefined label (last bar) or too-sparse features (warmup).
    coverage = X.notna().mean(axis=1)
    keep = y.notna() & (coverage >= _MIN_FEATURE_COVERAGE)
    X, y = X.loc[keep], y.loc[keep].astype(int)
    log.info("Prepared X=%s, y base rate=%.3f for %s.", X.shape, y.mean(), ticker)
    return X, y


def walk_forward_evaluate(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict[str, Any]:
    """Run walk-forward; return per-fold + aggregate classification metrics."""
    splits = walk_forward_splits(len(X), n_splits=n_splits)
    fold_metrics = []
    for sp in splits:
        clf = DownsideClassifier()
        clf.fit(X.iloc[sp.train_idx], y.iloc[sp.train_idx])
        p = clf.predict_proba(X.iloc[sp.test_idx])
        m = classification_metrics(y.iloc[sp.test_idx].values, p)
        m["fold"] = sp.fold
        fold_metrics.append(m)
        log.info("Fold %d: n=%d auc=%.3f brier=%.3f base=%.3f",
                 sp.fold, m["n"], m.get("auc", float("nan")), m["brier"], m["base_rate"])
    return {"folds": fold_metrics, "aggregate": aggregate_fold_metrics(fold_metrics)}


def train_final(X: pd.DataFrame, y: pd.Series, ticker: str, save: bool = True) -> DownsideClassifier:
    """Train on all available history and persist."""
    clf = DownsideClassifier().fit(X, y)
    if save:
        clf.save(model_path(ticker))
    return clf


def load_or_train(collected: dict[str, Any], ticker: str) -> DownsideClassifier:
    """Load a saved classifier, or train + save one if none exists."""
    path = model_path(ticker).with_suffix(".json")
    if path.exists():
        try:
            return DownsideClassifier.load(model_path(ticker))
        except Exception as exc:
            log.warning("Failed to load model (%s); retraining.", exc)
    X, y = prepare_xy(collected, ticker)
    return train_final(X, y, ticker)


# ---------------------------------------------------------------------------
# Council 2 orchestration: all three models (Phase 5)
# ---------------------------------------------------------------------------
def quantile_model_path(ticker: str):
    return project_root() / "models_store" / f"{ticker.upper()}_quantile"


def prepare_training_data(collected: dict[str, Any], ticker: str):
    """Return (X, y_label, y_future_return) sharing one aligned, masked index."""
    prices = collected["prices"]
    table = build_feature_table(prices, collected.get("context", {}), ticker)
    targets = build_targets(prices)
    cols = [c for c in CORE_FEATURE_COLS if c in table.columns]
    X = table[cols].copy()
    y_label = targets["downside_label"].reindex(X.index)
    y_future = targets["future_return"].reindex(X.index)

    coverage = X.notna().mean(axis=1)
    keep = y_label.notna() & y_future.notna() & (coverage >= _MIN_FEATURE_COVERAGE)
    return X.loc[keep], y_label.loc[keep].astype(int), y_future.loc[keep]


def train_all_models(collected: dict[str, Any], ticker: str, save: bool = True) -> dict[str, Any]:
    """Train classifier + quantile model; persist both. (GARCH is fit on demand.)"""
    X, y_label, y_future = prepare_training_data(collected, ticker)
    clf = DownsideClassifier().fit(X, y_label)
    qm = QuantileDownsideModel().fit(X, y_future)
    if save:
        clf.save(model_path(ticker))
        qm.save(quantile_model_path(ticker))
    return {"classifier": clf, "quantile": qm}


def _load_or_train_all(collected: dict[str, Any], ticker: str) -> dict[str, Any]:
    clf_path = model_path(ticker).with_suffix(".json")
    qm_path = quantile_model_path(ticker).with_suffix(".joblib")
    if clf_path.exists() and qm_path.exists():
        try:
            return {
                "classifier": DownsideClassifier.load(model_path(ticker)),
                "quantile": QuantileDownsideModel.load(quantile_model_path(ticker)),
            }
        except Exception as exc:
            log.warning("Failed to load models (%s); retraining.", exc)
    return train_all_models(collected, ticker)


def run_prediction_models(
    collected: dict[str, Any],
    ticker: str,
    as_of=None,
) -> dict[str, Any]:
    """Run Council 2: produce p_downside, GARCH volatility risk, and tail quantiles.

    Returns a dict with all three model outputs + the live feature row. This is
    the input to the Final Decision Engine (Phase 6).
    """
    models = _load_or_train_all(collected, ticker)
    clf: DownsideClassifier = models["classifier"]
    qm: QuantileDownsideModel = models["quantile"]

    row = build_live_feature_row(collected, ticker, as_of=as_of)

    # Model 1: downside classifier.
    p_downside = clf.predict_one(row)

    # Model 2: GARCH volatility / VaR / ES on historical log returns.
    close = collected["prices"].sort_index()["close"].astype(float)
    log_ret = np.log(close / close.shift(1))
    garch = fit_and_forecast(log_ret, vol="Garch", alpha=0.05)

    # Model 3: quantile downside-tail.
    tail = qm.predict_one(row)

    return {
        "ticker": ticker.upper(),
        "timestamp": row.get("timestamp"),
        "feature_row": row,
        "classifier": {
            "p_downside": p_downside,
            "xgb_confidence": abs(p_downside - 0.5) * 2.0,
            "top_features": clf.top_features(8),
        },
        "garch": garch,
        "quantile": tail,
    }
