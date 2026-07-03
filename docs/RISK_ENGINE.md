# DownsideIQ — Risk Engine

The risk engine (`services/risk_engine_service.py`) turns ensemble scores into a
governed decision: **SHORT / WATCH / NO TRADE**, with full justification.

## Threshold modes (spec §13) — never mixed
- **strict** (`config/thresholds.yaml`): fixed institutional thresholds for live/
  capital decisions. SHORT requires ALL of: `adjusted_p ≥ 0.65`, `agreement ≥
  0.70`, `data_confidence ≥ 0.75`, `price_volume_confirmation > 0.10`,
  `expected_edge ≥ min_edge`, not in earnings blackout, kill switch clear.
- **research**: percentile-calibrated. SHORT if `adjusted_risk` is in the top
  `short_percentile` of a trailing window of research-mode signals; WATCH for the
  next band. Exploratory only — tagged `mode='research'` in the DB; never counted
  as official performance.

News-only signals without price/volume confirmation → **WATCH, never SHORT**.

## Risk limits & kill switch (spec §14)
From `config/risk_limits.yaml`: max risk/trade, daily-loss, weekly-drawdown,
consecutive-loss cooldown, max open positions. The kill switch
(`risk/kill_switch.py`) reads realised paper trades; when tripped it **downgrades
SHORT → NO TRADE** and writes a `risk_events` row.

## Position sizing (spec §14.1)
Volatility-aware: `stop = max(floor, vol_mult·predicted_vol)`,
`size = account·risk_per_trade / stop_distance`, scaled by confidence ×
agreement × data_confidence. Short stop sits above entry, take-profit below.

## Governance record (spec §12)
Every signal persists a `governance` JSON with: decision, mode, threshold mode,
reason, all scores, news catalysts, gates (thresholds + pass/fail), kill-switch
status, data-quality status + reasons, position sizing, and top drivers — so the
system can always explain *why* it chose SHORT/WATCH/NO TRADE.

## Safety
`LIVE_TRADING_ENABLED` must be false; the API refuses to start otherwise.
Paper trading only.
