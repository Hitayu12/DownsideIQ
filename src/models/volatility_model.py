"""GARCH-family volatility model (Council 2, Model 2 — spec §10.2).

Answers: "How volatile is the stock likely to be over the next bar, and is
downside volatility elevated?" Fits a GARCH/EGARCH/GJR-GARCH on historical log
returns and produces a 1-step volatility forecast plus parametric VaR and
expected shortfall — the risk-side complement to the directional classifier.

Returns are scaled ×100 before fitting (the ``arch`` package optimises better
on percent returns) and forecasts are scaled back to return units.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logging_utils import get_logger

log = get_logger("models.volatility")

_SCALE = 100.0


class VolatilityModel:
    """Wraps arch_model with a simple forecast/VaR/ES interface."""

    def __init__(self, vol: str = "Garch", p: int = 1, q: int = 1, dist: str = "t"):
        self.vol, self.p, self.q, self.dist = vol, p, q, dist
        self.res = None
        self._baseline_sigma: float | None = None

    def fit(self, log_returns: pd.Series) -> "VolatilityModel":
        from arch import arch_model

        r = log_returns.dropna().astype(float) * _SCALE
        if len(r) < 50:
            log.warning("Only %d returns for GARCH fit; results may be unstable.", len(r))
        # EGARCH uses o (asymmetry); GJR via vol='Garch', o=1.
        kwargs: dict[str, Any] = dict(mean="Constant", vol=self.vol, p=self.p, q=self.q, dist=self.dist)
        if self.vol.lower() in ("egarch",) or self.vol.lower() == "garch":
            kwargs["o"] = 1 if self.vol.lower() == "egarch" else 0
        am = arch_model(r, **kwargs)
        self.res = am.fit(disp="off")
        self._baseline_sigma = float(np.sqrt(self.res.conditional_volatility.var()) or 1.0)
        self._cond_vol = self.res.conditional_volatility / _SCALE
        return self

    def forecast(self, alpha: float = 0.05) -> dict[str, float]:
        """1-step volatility forecast + parametric VaR/ES (return units, positive=loss)."""
        if self.res is None:
            raise RuntimeError("VolatilityModel not fitted.")
        fc = self.res.forecast(horizon=1, reindex=False)
        var_pct = float(fc.variance.values[-1, 0])      # in (pct)^2
        sigma = math.sqrt(var_pct) / _SCALE             # back to return units
        mu = float(self.res.params.get("mu", 0.0)) / _SCALE

        # Quantile of the fitted distribution.
        nu = float(self.res.params.get("nu", np.nan))
        if self.dist == "t" and nu == nu and nu > 2:
            z = stats.t.ppf(alpha, df=nu) * math.sqrt((nu - 2) / nu)
            # Expected shortfall for Student-t (standardised).
            pdf = stats.t.pdf(stats.t.ppf(alpha, df=nu), df=nu)
            es_factor = -(pdf / alpha) * ((nu + stats.t.ppf(alpha, df=nu) ** 2) / (nu - 1)) \
                * math.sqrt((nu - 2) / nu)
        else:
            z = stats.norm.ppf(alpha)
            es_factor = -stats.norm.pdf(stats.norm.ppf(alpha)) / alpha

        var_estimate = -(mu + sigma * z)                # positive loss magnitude
        es_estimate = -(mu + sigma * es_factor)

        # Regime + downside-vol-risk vs recent typical conditional vol.
        recent = float(self._cond_vol.tail(60).median()) if hasattr(self, "_cond_vol") else sigma
        ratio = sigma / recent if recent > 0 else 1.0
        regime = "high" if ratio > 1.25 else ("low" if ratio < 0.8 else "normal")

        return {
            "forecast_volatility": float(sigma),
            "volatility_ratio": float(ratio),
            "volatility_regime": regime,
            "var_estimate": float(var_estimate),
            "expected_shortfall_estimate": float(es_estimate),
            "downside_volatility_risk": float(np.clip((ratio - 0.8) / 0.7, 0, 1)),
            "garch_nu": nu if nu == nu else None,
        }


def fit_and_forecast(log_returns: pd.Series, vol: str = "Garch", alpha: float = 0.05) -> dict[str, float]:
    """Convenience: fit a GARCH model and return its 1-step forecast dict."""
    try:
        return VolatilityModel(vol=vol).fit(log_returns).forecast(alpha=alpha)
    except Exception as exc:
        log.warning("GARCH fit/forecast failed (%s); returning empirical fallback.", exc)
        r = log_returns.dropna()
        sigma = float(r.tail(20).std()) if len(r) else float("nan")
        return {
            "forecast_volatility": sigma,
            "volatility_ratio": 1.0,
            "volatility_regime": "normal",
            "var_estimate": float(-stats.norm.ppf(alpha) * sigma) if sigma == sigma else float("nan"),
            "expected_shortfall_estimate": float(stats.norm.pdf(stats.norm.ppf(alpha)) / alpha * sigma)
            if sigma == sigma else float("nan"),
            "downside_volatility_risk": 0.5,
            "garch_nu": None,
        }
