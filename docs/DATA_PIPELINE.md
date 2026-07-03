# DownsideIQ — Data Pipeline

## Providers (`src/providers/`)
| Provider | Role | Failure rule |
|---|---|---|
| yfinance | OHLCV (primary) | **Hard**: empty/stale/future/null → `DataQualityError`, signal blocked |
| Tavily | live news search | Soft: price-only mode, `news_confidence=0` |
| Gemini | structured news scoring | Soft: heuristic fallback; circuit breaker on quota |
| Alpha Vantage | fundamentals/earnings | Soft: skip fundamentals |
| Finnhub | company news, financials, earnings calendar | Soft: skip gracefully |

Every call: timeout + `tenacity` exponential-backoff retry (from
`config/data_sources.yaml`) + structured error log + typed fallback. No silent
failures; degradation is recorded on `DegradedMode` and surfaced in the signal's
`data_quality`.

## Flow
1. **Ingestion** (`ingestion_service`) pulls price (require=True), context assets,
   news (Tavily + Finnhub), fundamentals (AV + Finnhub); persists raw price/news.
2. **Feature** (`feature_service`) builds the leakage-safe snapshot
   (price/volume/volatility/market-context + timestamp-safe fundamentals),
   validates it, computes `data_confidence_score`, persists `feature_snapshots`.
3. **News scoring** (`news_scoring_service`) scores the top-N items via Gemini
   (cost-capped) and the rest heuristically; persists `structured_news_scores`;
   produces overlay aggregates.

## Leakage discipline (enforced + tested)
- Features at bar *t* use only data ≤ *t* (backward-looking windows; no negative
  shift). Proven by `tests/test_no_leakage.py` (features invariant to appended
  future bars).
- The only future data touched is the **label** (next-session return), built in
  `target_builder` after the prediction is logged.
- Quarterly fundamentals are suppressed unless released before the prediction's
  `as_of` (`fundamental_features._released_before`).
- News is a **live overlay**, never backfilled into training — Tavily cannot
  reconstruct leakage-free historical news.

## Data validation (blocks, not fakes)
Before inference: non-null close, no future timestamp, not stale (configurable),
valid `current_price`, feature completeness ≥ threshold → otherwise the snapshot
raises `DataQualityError` and no signal is produced.
