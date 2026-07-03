"""Forward-return target construction (spec §10.1).

This is the ONLY module that intentionally uses future data — to build the
supervised LABEL, not a feature. The label for bar ``t`` is derived from the
return between ``t`` and ``t+1`` (next session for daily bars). Features at
``t`` remain strictly backward-looking, so the train/predict contract is:

    features(t)  [uses <= t]   →   predict downside over (t, t+1]

The downside label is volatility-adjusted (spec §10.1):
    downside_label = 1 if future_return < -max(floor_pct, vol_mult * recent_vol)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config_loader import get_settings
from src.utils.logging_utils import get_logger

log = get_logger("models.target")


def build_targets(prices: pd.DataFrame, settings: dict | None = None) -> pd.DataFrame:
    """Return a DataFrame with future_return, downside_threshold, downside_label.

    The final bar(s) have NaN future_return (no next session) and must be
    dropped before training.
    """
    if prices.empty:
        return pd.DataFrame()
    settings = settings or get_settings()
    tcfg = settings.get("target", {})
    floor_pct = float(tcfg.get("downside_floor_pct", 0.0075))
    vol_mult = float(tcfg.get("downside_vol_mult", 0.5))
    vol_window = int(tcfg.get("recent_vol_window", 20))

    close = prices.sort_index()["close"].astype(float)
    log_ret = np.log(close / close.shift(1))

    # Next-session forward log return (future — label only).
    future_return = np.log(close.shift(-1) / close)
    recent_vol = log_ret.rolling(vol_window).std()
    threshold = np.maximum(floor_pct, vol_mult * recent_vol)

    out = pd.DataFrame(index=close.index)
    out["future_return"] = future_return
    out["recent_volatility"] = recent_vol
    out["downside_threshold"] = threshold
    out["downside_label"] = (future_return < -threshold).astype("float")
    # Where future_return is NaN (last bar), label is undefined.
    out.loc[future_return.isna(), "downside_label"] = np.nan

    rate = out["downside_label"].mean()
    log.info("Built targets: %d labeled bars, downside base rate=%.3f.",
             int(out["downside_label"].notna().sum()), rate if rate == rate else float("nan"))
    return out
