"""Phase 7 tests: sizing, paper trade mechanics, ledger, kill switch."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.risk.kill_switch import apply_risk_controls, evaluate_kill_switch
from src.risk.position_sizing import compute_sizing
from src.trading import signal_logger
from src.trading.paper_trader import simulate_short_trade


def _decision(p=0.8, agree=0.8, dc=0.9):
    return {"adjusted_p_downside": p, "model_agreement_score": agree, "data_confidence_score": dc,
            "decision": "SHORT", "garch": {"forecast_volatility": 0.02}}


def _bars(prices):
    idx = pd.date_range("2026-01-02", periods=len(prices), freq="B", tz="UTC")
    return pd.DataFrame({"open": prices, "high": [p * 1.001 for p in prices],
                         "low": [p * 0.999 for p in prices], "close": prices}, index=idx)


def test_sizing_short_stop_above_tp_below():
    s = compute_sizing(_decision(), entry_price=100.0, predicted_volatility=0.02)
    assert s["stop_loss"] > 100.0          # short stop is above entry
    assert s["take_profit"] < 100.0        # short target is below
    assert s["position_size"] > 0


def test_paper_trade_profit_when_price_falls():
    s = compute_sizing(_decision(), entry_price=100.0, predicted_volatility=0.02)
    # Price drifts down but never hits TP -> horizon expiry, profit for a short.
    bars = _bars([99.5, 99.0, 98.8])
    t = simulate_short_trade("sig1", "NVDA", bars.index[0], s, bars, horizon_bars=1)
    assert t["pnl"] > 0
    assert t["result"] == "win"


def test_paper_trade_stop_loss_when_price_spikes():
    s = compute_sizing(_decision(), entry_price=100.0, predicted_volatility=0.02)
    spike = s["stop_loss"] * 1.01
    bars = _bars([spike, spike, spike])
    t = simulate_short_trade("sig2", "NVDA", bars.index[0], s, bars, horizon_bars=2)
    assert t["exit_reason"] == "stop_loss"
    assert t["pnl"] < 0


def test_ledger_log_and_update(tmp_path, monkeypatch):
    monkeypatch.setattr(signal_logger, "data_dir", lambda: tmp_path)
    decision = {
        "timestamp": "2026-05-20 00:00:00+00:00", "ticker": "NVDA",
        "p_downside": 0.7, "adjusted_p_downside": 0.72, "decision": "SHORT",
        "garch": {"forecast_volatility": 0.02, "volatility_regime": "normal",
                  "var_estimate": 0.03, "expected_shortfall_estimate": 0.04},
        "quantile": {"predicted_5pct_return": -0.03, "predicted_10pct_return": -0.02},
        "model_agreement_score": 0.75, "data_confidence_score": 0.8,
        "base_downside_risk_score": 0.5, "adjusted_downside_risk_score": 0.4,
        "uncertainty_penalty": 0.05, "price_volume_confirmation": 0.3,
        "news_shift_logodds": 0.1, "expected_edge_bps": 100,
        "top_drivers": [], "news_catalysts": {}, "market_regime": "neutral",
    }
    sid = signal_logger.log_prediction(decision)
    df = signal_logger.load_ledger()
    assert len(df) == 1 and df.iloc[0]["signal_id"] == sid

    # Prices with a drop after the signal -> SHORT call correct.
    idx = pd.date_range("2026-05-19", periods=4, freq="B", tz="UTC")
    prices = pd.DataFrame({"close": [100, 100, 95, 94.0]}, index=idx)
    res = signal_logger.update_actual_outcome(sid, prices, horizon_bars=1)
    assert res["actual_return_24h"] < 0
    assert res["prediction_correct"] is True


def test_kill_switch_blocks_short_on_consecutive_losses():
    trades = pd.DataFrame({
        "exit_time": pd.date_range("2026-05-28", periods=3, freq="D", tz="UTC").astype(str),
        "pnl": [-50.0, -60.0, -40.0],
        "result": ["loss", "loss", "loss"],
    })
    ks = evaluate_kill_switch(trades)
    assert ks["active"] is True

    decision = {"decision": "SHORT", "reasons": []}
    out = apply_risk_controls(decision, paper_trades=trades)
    assert out["decision"] == "NO TRADE"
