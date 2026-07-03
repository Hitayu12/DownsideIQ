"""Phase 6 tests: Final Decision Engine (ensemble + overlay + gate)."""
from __future__ import annotations

from src.models.ensemble_engine import (
    decide,
    model_agreement,
    price_volume_confirmation,
)


def _outputs(*, p=0.8, dvr=0.8, tail=0.8, q5=-0.05, row=None):
    base_row = {
        "data_confidence_score": 0.9,
        "missing_data_flag": False,
        "earnings_date_distance_days": 60,
        "market_regime": "neutral",
        # price/volume confirmation inputs (strong downside confirmation):
        "abnormal_volume_score": 0.9, "momentum_score": -0.05,
        "relative_strength_vs_sector": -0.05, "volatility_ratio": 1.3,
        "overnight_gap": -0.02, "return_vs_peers_1b": -0.02,
        "correlation_spike_score": 0.2, "drawdown_from_recent_high": -0.1,
        # news (neutral by default):
        "company_news_risk_score": 0.0, "macro_risk_score": 0.0,
        "negative_catalyst_score": 0.0, "positive_catalyst_score": 0.0,
        "abnormal_news_volume_flag": 0.0,
    }
    if row:
        base_row.update(row)
    return {
        "ticker": "TEST", "timestamp": "2026-05-29",
        "feature_row": base_row,
        "classifier": {"p_downside": p, "top_features": [("momentum_score", 0.1)]},
        "garch": {"downside_volatility_risk": dvr, "forecast_volatility": 0.03,
                  "var_estimate": 0.05, "expected_shortfall_estimate": 0.07},
        "quantile": {"downside_tail_score": tail, "predicted_5pct_return": q5,
                     "predicted_10pct_return": q5 + 0.005},
    }


def test_strong_aligned_signal_shorts():
    d = decide(_outputs(p=0.85, dvr=0.8, tail=0.82))
    assert d["decision"] == "SHORT"
    assert d["model_agreement_score"] >= 0.7
    assert d["price_volume_confirmation"] > 0.1


def test_weak_signal_no_trade():
    weak_row = {"abnormal_volume_score": 0.0, "momentum_score": 0.05,
                "relative_strength_vs_sector": 0.05, "volatility_ratio": 0.9,
                "overnight_gap": 0.01, "return_vs_peers_1b": 0.01,
                "correlation_spike_score": 0.0, "drawdown_from_recent_high": 0.0}
    d = decide(_outputs(p=0.30, dvr=0.2, tail=0.2, q5=-0.005, row=weak_row))
    assert d["decision"] == "NO TRADE"


def test_news_only_without_confirmation_is_watch_not_short():
    """Strong bearish news but NO price/volume confirmation must not SHORT."""
    row = {
        "company_news_risk_score": 0.9, "macro_risk_score": 0.6,
        "negative_catalyst_score": 0.8,
        # neutralise price/volume confirmation:
        "abnormal_volume_score": 0.0, "momentum_score": 0.0,
        "relative_strength_vs_sector": 0.0, "volatility_ratio": 1.0,
        "overnight_gap": 0.0, "return_vs_peers_1b": 0.0,
        "correlation_spike_score": 0.0, "drawdown_from_recent_high": 0.0,
    }
    d = decide(_outputs(p=0.55, dvr=0.3, tail=0.3, row=row))
    assert d["decision"] != "SHORT"
    assert d["decision"] == "WATCH"


def test_news_shift_is_capped():
    row = {"company_news_risk_score": 1.0, "macro_risk_score": 1.0,
           "negative_catalyst_score": 1.0, "positive_catalyst_score": 0.0}
    d = decide(_outputs(p=0.5, row=row))
    assert d["news_shift_capped"] is True
    assert abs(d["news_shift_logodds"]) <= 1.2 + 1e-9


def test_earnings_blackout_blocks_short():
    d = decide(_outputs(p=0.9, dvr=0.85, tail=0.85, row={"earnings_date_distance_days": 0}))
    assert d["decision"] != "SHORT"


def test_missing_data_forces_no_trade():
    d = decide(_outputs(p=0.9, dvr=0.85, tail=0.85, row={"missing_data_flag": True}))
    assert d["decision"] == "NO TRADE"


def test_price_volume_confirmation_sign():
    bullish_row = {"abnormal_volume_score": 0.0, "momentum_score": 0.05,
                   "relative_strength_vs_sector": 0.05, "volatility_ratio": 0.8,
                   "overnight_gap": 0.02, "return_vs_peers_1b": 0.02,
                   "correlation_spike_score": 0.0, "drawdown_from_recent_high": 0.0}
    assert price_volume_confirmation(bullish_row) < 0.1


def test_model_agreement_bounds():
    assert model_agreement(0.8, 0.8, 0.8) > 0.9      # aligned
    assert model_agreement(0.9, 0.1, 0.5) < 0.6      # dispersed
