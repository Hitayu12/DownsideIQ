"""Alpha Vantage fundamentals provider.

Failure rule (spec §6): on failure, skip fundamentals gracefully (return {}),
never raise. Alpha Vantage signals rate limits inside a 200 response, which we
detect and treat as 'no usable payload'.
"""
from __future__ import annotations

from typing import Any

import requests

from src.core.config import env
from src.core.logging import get_logger
from src.providers.base import call_with_retry, provider_cfg

log = get_logger("providers.alpha_vantage")
_BASE = "https://www.alphavantage.co/query"


def available() -> bool:
    return bool(env().alpha_vantage_api_key)


def query(params: dict[str, Any]) -> dict[str, Any] | None:
    """Call Alpha Vantage; return parsed JSON or None (degraded) on failure."""
    e = env()
    if not e.alpha_vantage_api_key:
        return None
    cfg = provider_cfg("fundamentals")
    q = {**params, "apikey": e.alpha_vantage_api_key}

    def _get():
        resp = requests.get(_BASE, params=q, timeout=int(cfg.get("timeout_seconds", 20)))
        resp.raise_for_status()
        return resp.json()

    try:
        data = call_with_retry(_get, provider="alpha_vantage",
                               retries=int(cfg.get("retries", 2)),
                               backoff=float(cfg.get("retry_backoff_seconds", 3)))
    except Exception as exc:
        log.warning("alpha_vantage_failed", function=params.get("function"), error=str(exc)[:160])
        return None

    for flag in ("Note", "Information", "Error Message"):
        if flag in data:
            log.warning("alpha_vantage_notice", flag=flag, detail=str(data[flag])[:120])
            if flag in ("Note", "Error Message"):
                return None
    return data
