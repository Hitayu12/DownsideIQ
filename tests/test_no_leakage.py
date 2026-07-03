"""No-look-ahead-bias tests (spec §29 — the most important test file).

The central guarantee of DownsideIQ: a feature computed at bar ``t`` must depend
ONLY on data from bars ``<= t``. We prove this structurally: features computed
on history truncated at ``t`` must equal the same rows computed on the FULL
history. If any feature peeked at future bars, appending future data would
change past values and these tests would fail.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.market_context_features import compute_market_context_features
from src.features.price_features import compute_price_features
from src.features.volatility_features import compute_volatility_features
from src.utils.timestamp_utils import (
    LookAheadError,
    assert_no_future_data,
    filter_strictly_before,
)


def _synthetic_prices(n: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    rets = rng.normal(0.0005, 0.02, n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.01, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "adj_close": close, "volume": volume},
        index=idx,
    )


@pytest.mark.parametrize("builder", [compute_price_features, compute_volatility_features])
def test_features_invariant_to_future_data(builder):
    """Past feature values must not change when future bars are added."""
    prices = _synthetic_prices(300)
    cutoff = 200

    full = builder(prices)
    truncated = builder(prices.iloc[:cutoff])

    common = truncated.index
    full_slice = full.loc[common].select_dtypes(include=[np.number])
    trunc_slice = truncated.select_dtypes(include=[np.number])

    pd.testing.assert_frame_equal(
        full_slice, trunc_slice[full_slice.columns], check_exact=False, rtol=1e-9, atol=1e-12
    )


def test_market_context_invariant_to_future_data():
    prices = _synthetic_prices(300, seed=1)
    spy = _synthetic_prices(300, seed=2)
    smh = _synthetic_prices(300, seed=3)
    cfg = {"market_etfs": ["SPY"], "sector_etfs": ["SMH"], "peers": [], "vol_proxy": "^VIX"}

    full = compute_market_context_features(prices, {"SPY": spy, "SMH": smh}, cfg)
    cut = 220
    trunc = compute_market_context_features(
        prices.iloc[:cut], {"SPY": spy.iloc[:cut], "SMH": smh.iloc[:cut]}, cfg
    )
    cols = trunc.select_dtypes(include=[np.number]).columns
    pd.testing.assert_frame_equal(
        full.loc[trunc.index, cols], trunc[cols], check_exact=False, rtol=1e-9, atol=1e-12
    )


def test_no_negative_shift_in_source():
    """Guard against accidentally introducing a forward shift (look-ahead)."""
    import inspect

    import src.features.price_features as pf
    import src.features.volatility_features as vf
    import src.features.market_context_features as mf

    for mod in (pf, vf, mf):
        src = inspect.getsource(mod)
        assert "shift(-" not in src, f"Forward shift (look-ahead) found in {mod.__name__}"


def test_assert_no_future_data_guard():
    idx = pd.date_range("2026-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame({"x": range(5)}, index=idx)
    # Cutoff after all data -> fine.
    assert_no_future_data(df, "2026-01-10")
    # Cutoff inside the data -> must raise (a row at/after cutoff exists).
    with pytest.raises(LookAheadError):
        assert_no_future_data(df, "2026-01-03")


def test_filter_strictly_before():
    idx = pd.date_range("2026-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame({"x": range(5)}, index=idx)
    kept = filter_strictly_before(df, "2026-01-03")
    # Strictly before 2026-01-03 -> 2026-01-01, 2026-01-02 only.
    assert len(kept) == 2
    assert kept.index.max() < pd.Timestamp("2026-01-03", tz="UTC")
