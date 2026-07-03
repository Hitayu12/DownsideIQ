"""Volatility features (spec §8.3, §9).

Backward-looking realized/rolling volatility and a volatility-regime label
derived from the current vol relative to its own trailing distribution. These
feed both the feature row and the volatility-adjusted downside target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Annualisation is intentionally NOT applied — we model short-horizon (per-bar)
# volatility. Downstream code interprets these as per-bar return std-devs.


def compute_volatility_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Return rolling/realized volatility + regime features for ``prices``."""
    if prices.empty:
        return pd.DataFrame()

    df = prices.sort_index()
    close = df["close"].astype(float)
    log_ret = np.log(close / close.shift(1))

    out = pd.DataFrame(index=df.index)
    out["rolling_volatility_12b"] = log_ret.rolling(12).std()
    out["rolling_volatility_24b"] = log_ret.rolling(24).std()
    # Realized vol over a short recent window (sqrt of sum of squared returns).
    out["realized_volatility"] = np.sqrt((log_ret**2).rolling(10).sum())
    out["downside_volatility"] = log_ret.where(log_ret < 0).rolling(24).std()

    # Regime: current 12b vol vs its trailing 120b median (ratio), plus a label.
    cur = out["rolling_volatility_12b"]
    baseline = cur.rolling(120, min_periods=20).median()
    ratio = cur / baseline.replace(0, np.nan)
    out["volatility_ratio"] = ratio
    out["volatility_regime"] = pd.cut(
        ratio,
        bins=[-np.inf, 0.8, 1.25, np.inf],
        labels=["low", "normal", "high"],
    ).astype("object")
    # Recent volatility used by the downside-target threshold (Phase 4).
    out["recent_volatility"] = log_ret.rolling(20).std()
    return out
