# DownsideIQ — API

FastAPI backend. Start: `python -m src.cli run-api` (or via docker compose).
Base URL: `http://localhost:8000`. Interactive docs at `/docs`. No auth
(localhost research tool; auth is a roadmap item).

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Provider availability, DB status, last-signal time |
| POST | `/analyze/{ticker}?mode=strict\|research` | Enqueue analysis → `{job_id}` |
| GET | `/jobs/{job_id}` | Poll job: `{state, ready, result}` |
| GET | `/signals/latest?ticker=&mode=` | Latest signal (governed) |
| GET | `/signals/history?ticker=&mode=&limit=` | Signal history |
| GET | `/predictions/{signal_id}` | Full signal + governance record |
| GET | `/risk/status?ticker=&mode=` | Drift, rolling accuracy, kill switch |
| GET | `/paper-trades?ticker=&mode=research` | Paper-trade performance + trades |
| GET | `/model/status?ticker=` | Registered model metrics + active mode |

## Async pattern
```
POST /analyze/NVDA?mode=strict        -> {"job_id": "...", "status": "queued"}
GET  /jobs/<job_id>                    -> {"state": "PENDING"|"SUCCESS", "ready": bool, "result": {...}}
```
Work runs on the Celery worker (Redis broker). For local dev without Redis, set
`CELERY_EAGER=true` to run inline.

## Example
```bash
curl -s http://localhost:8000/health
JOB=$(curl -s -X POST "http://localhost:8000/analyze/NVDA?mode=strict" | jq -r .job_id)
curl -s "http://localhost:8000/jobs/$JOB" | jq .result
```

## Signal `result` shape (abridged)
```json
{
  "signal_id": "…", "ticker": "NVDA", "decision": "WATCH",
  "adjusted_p_downside": 0.61, "adjusted_downside_risk_score": 0.09,
  "data_quality": "ok",
  "reason": "p_downside 0.61 below 0.65; model agreement 0.66 below 0.70"
}
```
The full governance record (gates, scores, news catalysts, sizing, kill switch)
is available via `GET /predictions/{signal_id}`.
