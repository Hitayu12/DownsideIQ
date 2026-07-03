"""API response schemas (typed contracts for the HTTP layer)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    database: bool
    providers: dict[str, bool]
    last_signal_at: str | None = None


class JobResponse(BaseModel):
    job_id: str
    status: str
    mode: str
    ticker: str


class JobResult(BaseModel):
    job_id: str
    state: str
    ready: bool
    result: dict[str, Any] | None = None
