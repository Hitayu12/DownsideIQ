"""DownsideIQ command-line entrypoint.

Usage:
    python main.py --status
    python main.py --ticker NVDA --mode collect
    python main.py --ticker NVDA --mode features
    python main.py --ticker NVDA --mode predict
    python main.py --ticker NVDA --mode paper

Phase 1 implements the skeleton + `status`. Later modes are wired in as each
build phase lands; until then they fail gracefully with a clear message.
"""
from __future__ import annotations

import argparse
import sys

from src.utils.config_loader import (
    api_key_status,
    get_risk_limits,
    get_settings,
    get_ticker_config,
)
from src.utils.logging_utils import get_logger

log = get_logger("main")

MODES = ("collect", "features", "predict", "paper")


def cmd_status() -> int:
    """Print configuration + integration availability. Always safe to run."""
    settings = get_settings()
    risk = get_risk_limits()
    keys = api_key_status()

    print("=" * 60)
    print("  DownsideIQ — system status")
    print("=" * 60)
    print(f"  Default ticker      : {settings.get('default_ticker')}")
    print(f"  Horizon             : {settings.get('default_prediction_horizon')}")
    print(f"  Bar size            : {settings.get('bar_size')}")
    print(f"  Target              : {settings.get('target', {}).get('type')}")
    print(f"  News scorer         : {settings.get('news_scorer')}")
    print(f"  Paper trading mode  : {risk.get('paper_trading_mode')}")
    print(f"  LIVE trading enabled: {risk.get('live_trading_enabled')}  (must be False)")
    print("-" * 60)
    print("  Optional integrations (graceful-degrade if missing):")
    for name, available in keys.items():
        mark = "✓" if available else "·"
        state = "available" if available else "missing  -> fallback/mock"
        print(f"    [{mark}] {name:<14}: {state}")
    print("=" * 60)

    if risk.get("live_trading_enabled"):
        log.error("LIVE_TRADING_ENABLED is True — this must be False by default.")
        return 1
    return 0


def cmd_collect(ticker: str) -> int:
    """Phase 2: run the Data Intelligence Council and save raw data."""
    from src.pipeline import collect_data

    result = collect_data(ticker)
    m = result["manifest"]
    print("-" * 60)
    print(f"  Collected data for {m['ticker']}:")
    print(f"    price bars         : {m['price_bars']} (last {m['price_last_ts']})")
    print(f"    context assets     : {m['context_asset_count']} {m['context_assets']}")
    print(f"    macro events       : {m['macro_events']}")
    print(f"    company events     : {m['company_events']} (volume={m['company_news_volume']})")
    print(f"    next earnings      : {m['next_earnings_date']} "
          f"({m['earnings_date_distance_days']} days)")
    print(f"    fundamentals avail : {m['fundamentals_available']}")
    print("-" * 60)
    if m["price_bars"] == 0:
        log.error("No price data collected — check network/yfinance.")
        return 1
    return 0


def cmd_features(ticker: str) -> int:
    """Phase 3: build the feature table + live feature row."""
    from src.pipeline import build_features

    result = build_features(ticker)
    table, row = result["table"], result["row"]
    if table.empty:
        log.error("Feature table is empty — collect data first (--mode collect).")
        return 1
    print("-" * 60)
    print(f"  Feature table for {ticker}: {len(table)} rows × {table.shape[1]} cols")
    print(f"  Live feature row @ {row.get('timestamp')}:")
    highlights = [
        "current_price", "return_1b", "momentum_score", "rolling_volatility_12b",
        "volatility_regime", "market_beta", "stock_specific_move_score",
        "company_news_risk_score", "macro_risk_score", "negative_catalyst_score",
        "earnings_date_distance_days", "fundamental_risk_score",
        "data_confidence_score", "missing_data_flag",
    ]
    for k in highlights:
        if k in row:
            v = row[k]
            v = round(v, 4) if isinstance(v, float) else v
            print(f"    {k:28}: {v}")
    print("-" * 60)
    return 0


def cmd_predict(ticker: str) -> int:
    """Phase 6: run Council 2 + Final Decision Engine and print the decision."""
    from src.pipeline import generate_decision

    d = generate_decision(ticker)
    badge = {"SHORT": "🔴 SHORT", "WATCH": "🟡 WATCH", "NO TRADE": "⚪ NO TRADE"}.get(d["decision"], d["decision"])
    print("=" * 60)
    print(f"  {d['ticker']}  @ {d['timestamp']}   →   {badge}")
    print("=" * 60)
    print(f"  p_downside (model)   : {d['p_downside']:.3f}")
    print(f"  adjusted p_downside  : {d['adjusted_p_downside']:.3f}  "
          f"(news shift {d['news_shift_logodds']:+.2f} log-odds"
          f"{', CAPPED' if d['news_shift_capped'] else ''})")
    print(f"  adjusted risk score  : {d['adjusted_downside_risk_score']:.3f}")
    print(f"  model agreement      : {d['model_agreement_score']:.3f}")
    print(f"  data confidence      : {d['data_confidence_score']:.3f}")
    print(f"  price/vol confirm    : {d['price_volume_confirmation']:+.3f}")
    print(f"  regime adjustment    : {d['market_regime_adjustment']:.2f}  ({d['market_regime']})")
    print(f"  uncertainty penalty  : {d['uncertainty_penalty']:.3f}")
    print(f"  expected edge        : {d['expected_edge_bps']:.0f} bps")
    g = d["garch"]
    print(f"  GARCH vol / VaR5     : {g['forecast_volatility']:.4f} / {g['var_estimate']:.4f}")
    print(f"  quantile q5 / q10    : {d['quantile']['predicted_5pct_return']:.4f} / "
          f"{d['quantile']['predicted_10pct_return']:.4f}")
    print("  reasons:")
    for r in d["reasons"]:
        print(f"    • {r}")
    print("  top drivers:")
    for drv in d["top_drivers"]:
        print(f"    • {drv['driver']}: {drv['value']} (w={drv['weight']})")
    print("=" * 60)
    return 0


def cmd_paper(ticker: str) -> int:
    """Phase 7: strict honest backtest (ledger) + exploratory backtest (trades) + live log."""
    import json

    from src.pipeline import generate_decision, load_collected
    from src.risk.kill_switch import apply_risk_controls
    from src.risk.position_sizing import compute_sizing
    from src.trading.paper_backtest import EXPLORATORY_OVERRIDES, run_paper_backtest
    from src.trading.signal_logger import load_ledger, log_prediction
    from src.utils.config_loader import data_dir

    collected = load_collected(ticker)
    if collected is None:
        log.error("No collected data for %s — run --mode collect first.", ticker)
        return 1

    # 1) STRICT backtest -> honest signal-history ledger (live gate, no relaxed trades).
    strict = run_paper_backtest(ticker, collected, mode="strict", trade_decisions=(),
                                write_ledger=True, write_trades=False)

    # 2) EXPLORATORY backtest -> paper trades for mechanics/dashboard (clearly labelled).
    explor = run_paper_backtest(ticker, collected, mode="exploratory",
                                gate_overrides=EXPLORATORY_OVERRIDES, trade_decisions=("SHORT",),
                                write_ledger=False, write_trades=True)

    # 3) Live prediction (STRICT gate) -> risk controls -> append to ledger (outcome pending).
    decision = generate_decision(ticker, collected=collected)
    decision = apply_risk_controls(decision, paper_trades=None)
    sizing = None
    if decision["decision"] == "SHORT":
        entry = decision.get("feature_row", {}).get("current_price", 0.0)
        sizing = compute_sizing(decision, entry, decision["garch"]["forecast_volatility"])
    signal_id = log_prediction(decision, sizing)

    # Persist summaries for the dashboard.
    reports = data_dir() / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    with (reports / f"{ticker}_backtest_summary.json").open("w", encoding="utf-8") as fh:
        json.dump({"strict": strict, "exploratory": explor}, fh, indent=2, default=str)

    m = explor["trading_metrics"]
    print("=" * 64)
    print(f"  Paper trading — {ticker}")
    print("=" * 64)
    print(f"  Live signal (STRICT) : {signal_id} -> {decision['decision']}")
    print(f"  OOS window           : train={strict['train_bars']} test={strict['test_bars']} bars")
    print("  -- STRICT gate (the real/live discipline) --")
    print(f"    decisions          : {strict['decision_counts']}")
    print(f"    SHORTs             : {strict['decision_counts'].get('SHORT', 0)}")
    print(f"    NO-TRADE correct   : {strict['no_trade_correct_rate']:.2%}")
    print("  -- EXPLORATORY gate (relaxed; demo of trade mechanics only) --")
    print(f"    decisions          : {explor['decision_counts']}")
    print(f"    paper trades       : {explor['n_trades']}")
    if explor["n_trades"]:
        print(f"    hit rate           : {m['hit_rate']:.2%}")
        print(f"    profit factor      : {m['profit_factor']:.2f}")
        print(f"    total P&L          : ${m['total_pnl']:.2f}")
        print(f"    Sharpe / Sortino   : {m['sharpe_ratio']:.2f} / {m['sortino_ratio']:.2f}")
        print(f"    max drawdown ($)   : {m['max_drawdown']:.2f}")
        print(f"    false-positive short: {m.get('false_positive_short_rate', float('nan')):.2%}")
    print(f"  Ledger rows          : {len(load_ledger())}")
    print("=" * 64)
    return 0


def _not_yet(mode: str, ticker: str) -> int:
    log.warning(
        "Mode '%s' for %s is not implemented yet (lands in a later build phase). "
        "Run `python main.py --status` to verify the skeleton.",
        mode,
        ticker,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="downsideiq",
        description="DownsideIQ — live short-horizon equity downside-risk engine.",
    )
    parser.add_argument("--ticker", default=None, help="Ticker symbol, e.g. NVDA.")
    parser.add_argument("--mode", choices=MODES, default=None, help="Pipeline stage to run.")
    parser.add_argument("--status", action="store_true", help="Print system status and exit.")
    args = parser.parse_args(argv)

    if args.status or args.mode is None:
        return cmd_status()

    ticker = (args.ticker or get_settings().get("default_ticker", "NVDA")).upper()
    try:
        get_ticker_config(ticker)  # validate ticker is configured
    except KeyError as exc:
        log.error(str(exc))
        return 2

    log.info("Running mode=%s for ticker=%s", args.mode, ticker)
    if args.mode == "collect":
        return cmd_collect(ticker)
    if args.mode == "features":
        return cmd_features(ticker)
    if args.mode == "predict":
        return cmd_predict(ticker)
    if args.mode == "paper":
        return cmd_paper(ticker)
    return _not_yet(args.mode, ticker)


if __name__ == "__main__":
    sys.exit(main())
