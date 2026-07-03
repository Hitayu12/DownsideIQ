"""Price & volume features (Council 1 → Feature Engineering, spec §8.3, §9).

All features are strictly backward-looking: the value at bar ``t`` uses only
data from bars ``<= t``. Rolling windows and positive ``shift`` only — never a
negative shift (that would peek into the future). The forward-looking TARGET is
built separately in the model layer (Phase 4) and is the only place future data
is touched.

Windows are expressed in BARS so the same code works for daily (MVP) or
intraday bars. For daily data, 1 bar ≈ the next-session horizon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(a, b):
    return a / b.replace(0, np.nan)


def compute_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of price/volume features aligned to ``prices.index``.

    Expects columns: open, high, low, close, volume (adj_close optional).
    """
    if prices.empty:
        return pd.DataFrame()

    df = prices.sort_index().copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)
    volume = df["volume"].astype(float)
    prev_close = close.shift(1)

    out = pd.DataFrame(index=df.index)

    # --- Returns over bar windows (past returns; leakage-safe) ---
    out["log_return_1b"] = np.log(_safe_div(close, prev_close))
    for k in (1, 5, 10, 20):
        out[f"return_{k}b"] = close.pct_change(k)

    # --- Momentum & mean reversion ---
    ma_short, ma_long = close.rolling(10).mean(), close.rolling(50).mean()
    out["moving_average_spread"] = _safe_div(ma_short - ma_long, ma_long)
    out["momentum_score"] = close.pct_change(20)              # 20-bar momentum
    roll_mean = close.rolling(20).mean()
    roll_std = close.rolling(20).std()
    out["mean_reversion_score"] = -_safe_div(close - roll_mean, roll_std)  # z-score, inverted

    # --- Drawdown from recent high ---
    roll_max = close.rolling(20).max()
    out["drawdown_from_recent_high"] = _safe_div(close - roll_max, roll_max)

    # --- Intraday range & overnight gap ---
    out["intraday_range"] = _safe_div(high - low, close)
    gap = _safe_div(open_ - prev_close, prev_close)
    out["overnight_gap"] = gap
    out["gap_risk_score"] = gap.abs().rolling(20).mean()      # typical recent gap size

    # --- Volume ---
    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    out["volume_z_score"] = _safe_div(volume - vol_mean, vol_std)
    out["abnormal_volume_score"] = out["volume_z_score"].clip(lower=0) / 3.0  # 0..~1

    out["current_price"] = close
    return out
