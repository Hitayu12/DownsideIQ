"""Out-of-sample paper backtest (spec §16, §18) — the accountability loop.

Trains the classifier + quantile model on the EARLY part of history, then walks
the held-out bars: predict → decide → (if SHORT) size + simulate a paper trade
with the REALISED next-session outcome → log everything to the prediction ledger
and paper-trades file. This is genuinely out-of-sample (the model never saw the
test bars) and leakage-free (features are backward-looking; only the realised
label uses the future, after the prediction is logged).

News is absent historically (it cannot be backfilled leakage-free), so the live
news overlay contributes zero here — exactly matching the hybrid architecture.
Price/volume confirmation, which IS backfillable, still drives the gate.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.backtesting.attribution import run_attribution
from src.backtesting.metrics import trading_metrics
from src.features.feature_builder import CORE_FEATURE_COLS, build_feature_table
from src.models.downside_classifier import DownsideClassifier
from src.models.ensemble_engine import decide
from src.models.quantile_model import QuantileDownsideModel
from src.models.target_builder import build_targets
from src.trading.paper_trader import simulate_short_trade
from src.trading.signal_logger import LEDGER_COLUMNS, ledger_path
from src.utils.config_loader import data_dir, get_ticker_config
from src.utils.logging_utils import get_logger

log = get_logger("trading.backtest")

_NEWS_ZERO = {
    "company_news_risk_score": 0.0, "macro_risk_score": 0.0,
    "negative_catalyst_score": 0.0, "positive_catalyst_score": 0.0,
    "abnormal_news_volume_flag": 0.0, "macro_sentiment_score": 0.0,
}


def _garch_proxy(row: dict) -> dict[str, Any]:
    """Cheap GARCH stand-in for the backtest (avoids refitting per bar)."""
    vr = row.get("volatility_ratio", 1.0)
    vr = 1.0 if (vr is None or vr != vr) else float(vr)
    vol = row.get("rolling_volatility_12b", np.nan)
    vol = float(vol) if vol == vol else 0.02
    return {
        "forecast_volatility": vol,
        "volatility_ratio": vr,
        "volatility_regime": "high" if vr > 1.25 else ("low" if vr < 0.8 else "normal"),
        "var_estimate": 1.645 * vol,
        "expected_shortfall_estimate": 2.063 * vol,
        "downside_volatility_risk": float(np.clip((vr - 0.8) / 0.7, 0, 1)),
        "garch_nu": None,
    }


# Relaxed gate for the clearly-labeled EXPLORATORY backtest (NOT the live gate).
EXPLORATORY_OVERRIDES = {
    "minimum_downside_probability": 0.52,
    "minimum_model_agreement": 0.50,
    "minimum_data_confidence": 0.55,
    "pv_confirm_threshold": 0.0,
    "require_price_volume_confirmation": True,
}


def run_paper_backtest(
    ticker: str,
    collected: dict[str, Any],
    train_frac: float = 0.6,
    horizon_bars: int = 1,
    trade_decisions: tuple[str, ...] = ("SHORT",),
    gate_overrides: dict[str, Any] | None = None,
    mode: str = "strict",
    write_ledger: bool = True,
    write_trades: bool = True,
) -> dict[str, Any]:
    """Run the OOS paper backtest; persist ledger + paper trades; return summary.

    ``mode='strict'`` uses the live institutional gate (honest signal history).
    ``mode='exploratory'`` (with ``gate_overrides``) relaxes thresholds purely to
    demonstrate trade mechanics + populate the dashboard — clearly labelled.
    """
    ticker = ticker.upper()
    prices = collected["prices"].sort_index()
    context = collected.get("context", {})
    ticker_cfg = get_ticker_config(ticker)

    table = build_feature_table(prices, context, ticker)
    targets = build_targets(prices)
    cols = [c for c in CORE_FEATURE_COLS if c in table.columns]

    X = table[cols]
    y_label = targets["downside_label"].reindex(X.index)
    y_future = targets["future_return"].reindex(X.index)
    coverage = X.notna().mean(axis=1)
    keep = y_label.notna() & y_future.notna() & (coverage >= 0.5)
    X, y_label, y_future = X.loc[keep], y_label.loc[keep].astype(int), y_future.loc[keep]

    n = len(X)
    cut = int(n * train_frac)
    if cut < 60 or n - cut < 5:
        raise ValueError(f"Not enough data for backtest (n={n}).")

    clf = DownsideClassifier().fit(X.iloc[:cut], y_label.iloc[:cut])
    qm = QuantileDownsideModel().fit(X.iloc[:cut], y_future.iloc[:cut])

    test_idx = X.index[cut:]
    p_all = clf.predict_proba(X.loc[test_idx])
    q5_all = qm.models[0.05].predict(X.loc[test_idx].values)
    q10_all = qm.models[0.10].predict(X.loc[test_idx].values)

    ledger_rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []

    for i, ts in enumerate(test_idx):
        full_row = table.loc[ts].to_dict()
        row = {**full_row, **_NEWS_ZERO}
        core_present = np.mean([full_row.get(c) == full_row.get(c) for c in cols])  # non-NaN frac
        row["data_confidence_score"] = float(0.5 + 0.4 * core_present)  # no news historically
        row["missing_data_flag"] = bool(core_present < 0.8)
        row["earnings_date_distance_days"] = np.nan
        row["current_price"] = full_row.get("current_price")
        row["timestamp"] = str(ts)

        model_outputs = {
            "ticker": ticker, "timestamp": str(ts), "feature_row": row,
            "classifier": {"p_downside": float(p_all[i]), "top_features": []},
            "garch": _garch_proxy(row),
            "quantile": {"predicted_5pct_return": float(q5_all[i]),
                         "predicted_10pct_return": float(q10_all[i]),
                         "downside_tail_score": float(np.clip(-q5_all[i] / 0.05, 0, 1))},
        }
        d = decide(model_outputs, gate_overrides=gate_overrides)

        # Realised next-session outcome (future is known in backtest).
        entry_close = float(prices.loc[ts, "close"])
        forward = prices.loc[prices.index > ts]
        actual_return = (float(forward.iloc[horizon_bars - 1]["close"]) - entry_close) / entry_close \
            if len(forward) >= horizon_bars else np.nan
        correct = (actual_return < 0) if d["decision"] in ("SHORT", "WATCH") \
            else (actual_return >= -0.005) if actual_return == actual_return else None

        paper_result = None
        if d["decision"] in trade_decisions and len(forward) >= horizon_bars:
            from src.risk.position_sizing import compute_sizing
            sizing = compute_sizing(d, entry_close, row.get("rolling_volatility_12b", 0.02))
            trade = simulate_short_trade(
                signal_id=f"bt_{i:04d}", ticker=ticker, entry_time=ts,
                sizing=sizing, forward_bars=forward, horizon_bars=horizon_bars,
                market_regime=row.get("market_regime"),
            )
            attr = run_attribution(
                {**d, "market_beta": row.get("market_beta", 1.0)}, prices, context, ticker_cfg, horizon_bars
            )
            trade["attribution"] = attr.get("dominant_driver")
            trade["lesson"] = attr.get("lesson")
            trade["mode"] = mode
            trades.append(trade)
            paper_result = trade["result"]

        ledger_rows.append(_ledger_row(f"bt_{i:04d}", d, entry_close, actual_return, correct, paper_result))

    ledger_df = pd.DataFrame(ledger_rows).reindex(columns=LEDGER_COLUMNS)
    if write_ledger:
        ledger_df.to_csv(ledger_path(), index=False)

    trades_df = pd.DataFrame(trades)
    if write_trades:
        trades_path = data_dir() / "predictions" / "paper_trades.csv"
        trades_df.to_csv(trades_path, index=False)

    metrics = trading_metrics(trades_df) if not trades_df.empty else {"n_trades": 0}
    decision_counts = ledger_df["decision"].value_counts().to_dict()
    no_trade_correct = float(
        ledger_df.loc[ledger_df["decision"] == "NO TRADE", "prediction_correct"].mean()
    ) if (ledger_df["decision"] == "NO TRADE").any() else float("nan")

    summary = {
        "ticker": ticker,
        "mode": mode,
        "n_predictions": len(ledger_df),
        "decision_counts": decision_counts,
        "n_trades": metrics.get("n_trades", 0),
        "trading_metrics": metrics,
        "no_trade_correct_rate": no_trade_correct,
        "train_bars": cut,
        "test_bars": len(test_idx),
    }
    log.info("Paper backtest %s: %s decisions, %d trades, hit_rate=%.2f, PF=%.2f.",
             ticker, decision_counts, metrics.get("n_trades", 0),
             metrics.get("hit_rate", float("nan")), metrics.get("profit_factor", float("nan")))
    return summary


def _ledger_row(signal_id, d, entry_close, actual_return, correct, paper_result) -> dict[str, Any]:
    g, q = d["garch"], d["quantile"]
    return {
        "signal_id": signal_id, "timestamp": d["timestamp"], "ticker": d["ticker"],
        "current_price": entry_close, "prediction_horizon": "24h",
        "xgb_downside_probability": d["p_downside"],
        "adjusted_downside_probability": d["adjusted_p_downside"],
        "xgb_confidence": abs(d["adjusted_p_downside"] - 0.5) * 2,
        "garch_volatility_forecast": g.get("forecast_volatility"),
        "garch_volatility_regime": g.get("volatility_regime"),
        "var_estimate": g.get("var_estimate"),
        "expected_shortfall_estimate": g.get("expected_shortfall_estimate"),
        "quantile_5pct_return": q.get("predicted_5pct_return"),
        "quantile_10pct_return": q.get("predicted_10pct_return"),
        "model_agreement_score": d["model_agreement_score"],
        "data_confidence_score": d["data_confidence_score"],
        "base_downside_risk_score": d["base_downside_risk_score"],
        "adjusted_downside_risk_score": d["adjusted_downside_risk_score"],
        "uncertainty_penalty": d["uncertainty_penalty"],
        "price_volume_confirmation": d["price_volume_confirmation"],
        "news_shift_logodds": d["news_shift_logodds"],
        "expected_edge_bps": d["expected_edge_bps"],
        "decision": d["decision"],
        "recommended_position_size": np.nan, "position_notional": np.nan,
        "stop_loss": np.nan, "take_profit": np.nan,
        "top_3_signal_drivers": str(d.get("top_drivers", [])[:3]),
        "news_catalysts": str(d.get("news_catalysts", {})),
        "market_regime": d.get("market_regime"),
        "actual_return_12h": np.nan, "actual_return_24h": actual_return,
        "prediction_correct": correct, "paper_trade_result": paper_result,
        "post_trade_attribution": None,
        "created_at": d["timestamp"], "updated_at": d["timestamp"],
    }
