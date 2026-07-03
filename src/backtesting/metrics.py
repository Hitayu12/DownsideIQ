"""Evaluation metrics (spec §19).

Phase 4 implements the ML/classification metrics. Trading/risk metrics
(profit factor, Sharpe, Sortino, max drawdown, false-positive-short rate, …)
are added in the paper-trading / backtest phases.
"""
from __future__ import annotations

import numpy as np


def classification_metrics(y_true: np.ndarray, p_pred: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """AUC, precision/recall/F1, Brier, accuracy, base rate (NaN-safe)."""
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    p_pred = np.asarray(p_pred, dtype=float)
    mask = ~np.isnan(p_pred)
    y_true, p_pred = y_true[mask], p_pred[mask]
    y_hat = (p_pred >= threshold).astype(int)

    out: dict[str, float] = {
        "n": int(len(y_true)),
        "base_rate": float(y_true.mean()) if len(y_true) else float("nan"),
        "accuracy": float(accuracy_score(y_true, y_hat)) if len(y_true) else float("nan"),
        "brier": float(brier_score_loss(y_true, p_pred)) if len(y_true) else float("nan"),
    }
    # AUC requires both classes present.
    if len(np.unique(y_true)) == 2:
        out["auc"] = float(roc_auc_score(y_true, p_pred))
        out["precision"] = float(precision_score(y_true, y_hat, zero_division=0))
        out["recall"] = float(recall_score(y_true, y_hat, zero_division=0))
        out["f1"] = float(f1_score(y_true, y_hat, zero_division=0))
    else:
        out["auc"] = float("nan")
        out["precision"] = out["recall"] = out["f1"] = float("nan")
    return out


def trading_metrics(trades: "list[dict] | object", periods_per_year: int = 252) -> dict[str, float]:
    """Trading/risk metrics from a list of trade dicts (spec §19).

    Each trade needs at least ``return_pct``, ``pnl``, ``result``. Returns
    hit_rate, profit_factor, Sharpe, Sortino, max_drawdown, expectancy, etc.
    """
    import pandas as pd

    df = pd.DataFrame(list(trades)) if not isinstance(trades, pd.DataFrame) else trades
    if df.empty:
        return {"n_trades": 0}

    rets = df["return_pct"].astype(float).values
    pnl = df["pnl"].astype(float).values
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(equity) if len(equity) else np.array([0.0])
    drawdown = (equity - peak)
    mean_r, std_r = float(np.mean(rets)), float(np.std(rets))
    downside = rets[rets < 0]
    dstd = float(np.std(downside)) if len(downside) else 0.0

    out = {
        "n_trades": int(len(df)),
        "hit_rate": float((pnl > 0).mean()),
        "total_pnl": float(pnl.sum()),
        "average_return_per_signal": mean_r,
        "average_win": float(wins.mean()) if len(wins) else 0.0,
        "average_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0,
        "expectancy_per_trade": float(np.mean(pnl)),
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "sharpe_ratio": float(mean_r / std_r * np.sqrt(periods_per_year)) if std_r > 0 else 0.0,
        "sortino_ratio": float(mean_r / dstd * np.sqrt(periods_per_year)) if dstd > 0 else 0.0,
    }
    # False-positive short rate: shorts that ended in a loss.
    if "result" in df:
        shorts = df
        out["false_positive_short_rate"] = float((shorts["result"] == "loss").mean()) if len(shorts) else 0.0
    return out


def aggregate_fold_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Mean of each metric across walk-forward folds (ignoring NaNs)."""
    if not fold_metrics:
        return {}
    keys = set().union(*[m.keys() for m in fold_metrics])
    agg = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics if k in m and not np.isnan(m[k])]
        agg[f"{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
    return agg
