"""Paper trading service (spec §16) — simulate + record trades, compute perf.

Simulates short trades (reusing the validated simulator) and persists them to
``paper_trades`` tagged with ``mode`` (strict vs research) so official and
exploratory performance never mix. NO real-money execution.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.backtesting.metrics import trading_metrics
from src.core.logging import get_logger
from src.db import repositories as repo
from src.db.session import get_session
from src.trading.paper_trader import simulate_short_trade

log = get_logger("services.paper_trading")


class PaperTradingService:
    def simulate_and_record(self, *, signal_id: str | None, ticker: str, sizing: dict,
                            entry_time, forward_bars: pd.DataFrame, mode: str,
                            horizon_bars: int = 1, market_regime: str | None = None) -> dict[str, Any]:
        trade = simulate_short_trade(signal_id or "n/a", ticker, entry_time, sizing,
                                     forward_bars, horizon_bars, market_regime)
        trade["mode"] = mode
        self._persist(trade)
        log.info("paper_trade_recorded", trade_id=trade["trade_id"], ticker=ticker,
                 mode=mode, result=trade["result"], pnl=round(trade["pnl"], 2))
        return trade

    def _persist(self, trade: dict) -> None:
        import pandas as pd
        row = {k: trade.get(k) for k in (
            "trade_id", "signal_id", "ticker", "mode", "side", "entry_time", "entry_price",
            "position_size", "position_notional", "stop_loss", "take_profit", "exit_time",
            "exit_price", "exit_reason", "return_pct", "pnl", "result", "market_regime", "attribution")}
        for tcol in ("entry_time", "exit_time"):
            if isinstance(row.get(tcol), str):
                row[tcol] = pd.to_datetime(row[tcol], utc=True).to_pydatetime()
        with get_session() as s:
            repo.save_paper_trade(s, row)

    def performance(self, ticker: str | None = None, mode: str = "strict") -> dict[str, Any]:
        with get_session() as s:
            trades = repo.paper_trades(s, ticker=ticker, mode=mode)
        if not trades:
            return {"n_trades": 0, "mode": mode}
        df = pd.DataFrame([{"return_pct": t.return_pct, "pnl": t.pnl, "result": t.result,
                            "market_regime": t.market_regime} for t in trades
                           if t.pnl is not None])
        metrics = trading_metrics(df) if not df.empty else {"n_trades": 0}
        metrics["mode"] = mode
        return metrics

    def list_trades(self, ticker: str | None = None, mode: str | None = None) -> list[dict]:
        with get_session() as s:
            return [{
                "trade_id": t.trade_id, "signal_id": t.signal_id, "ticker": t.ticker,
                "mode": t.mode, "entry_time": str(t.entry_time), "entry_price": t.entry_price,
                "exit_time": str(t.exit_time), "exit_price": t.exit_price,
                "exit_reason": t.exit_reason, "return_pct": t.return_pct, "pnl": t.pnl,
                "result": t.result, "market_regime": t.market_regime, "attribution": t.attribution,
            } for t in repo.paper_trades(s, ticker=ticker, mode=mode)]
