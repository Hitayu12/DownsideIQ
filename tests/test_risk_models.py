"""Phase 5 tests: GARCH volatility model + quantile downside-tail model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.quantile_model import QuantileDownsideModel
from src.models.volatility_model import fit_and_forecast


def _garch_like_returns(n=800, seed=3):
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    sigma2 = 0.0004
    for t in range(1, n):
        sigma2 = 0.00001 + 0.08 * r[t - 1] ** 2 + 0.9 * sigma2
        r[t] = rng.normal(0, np.sqrt(sigma2))
    idx = pd.date_range("2022-01-01", periods=n, freq="B", tz="UTC")
    return pd.Series(r, index=idx)


def test_garch_forecast_sane():
    out = fit_and_forecast(_garch_like_returns(), vol="Garch", alpha=0.05)
    assert out["forecast_volatility"] > 0
    assert out["var_estimate"] > 0
    # Expected shortfall is at least as severe as VaR (further in the tail).
    assert out["expected_shortfall_estimate"] >= out["var_estimate"]
    assert out["volatility_regime"] in {"low", "normal", "high"}


def test_garch_fallback_on_tiny_data():
    # Too few points for a stable fit -> fallback path must still return finite vol.
    out = fit_and_forecast(pd.Series(np.random.default_rng(0).normal(0, 0.01, 10)))
    assert out["forecast_volatility"] == out["forecast_volatility"]   # not NaN
    assert "var_estimate" in out


def test_quantile_monotonic_and_negative_tail():
    rng = np.random.default_rng(1)
    n = 800
    feat = rng.normal(0, 1, n)
    # Future return depends on feature with heteroskedastic noise.
    future = -0.01 * np.clip(feat, 0, None) + rng.normal(0, 0.02, n)
    X = pd.DataFrame({"f": feat, "g": rng.normal(0, 1, n)})
    y = pd.Series(future)

    qm = QuantileDownsideModel(max_iter=120).fit(X.iloc[:600], y.iloc[:600])
    preds5 = qm.models[0.05].predict(X.iloc[600:].values)
    preds10 = qm.models[0.10].predict(X.iloc[600:].values)
    # On average the 5th percentile is below (more negative than) the 10th.
    assert preds5.mean() <= preds10.mean()

    out = qm.predict_one({"f": 2.0, "g": 0.0})
    assert out["predicted_5pct_return"] <= out["predicted_10pct_return"] + 1e-6
    assert 0.0 <= out["downside_tail_score"] <= 1.0


def test_quantile_save_load(tmp_path):
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"f": rng.normal(0, 1, 300), "g": rng.normal(0, 1, 300)})
    y = pd.Series(rng.normal(0, 0.02, 300))
    qm = QuantileDownsideModel(max_iter=60).fit(X, y)
    qm.save(tmp_path / "q")
    qm2 = QuantileDownsideModel.load(tmp_path / "q")
    a = qm.predict_one({"f": 1.0, "g": 0.0})
    b = qm2.predict_one({"f": 1.0, "g": 0.0})
    assert abs(a["predicted_5pct_return"] - b["predicted_5pct_return"]) < 1e-9
