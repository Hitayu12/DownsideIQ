"""Gemini LLM provider (structured generation) with a circuit breaker.

Failure rule (spec §6): on quota/permission/transient failure, fall back to the
heuristic scorer. The circuit breaker disables the LLM for the rest of the
process after the first hard failure to avoid hammering a dead quota.
``generate_json`` returns raw model text or ``None``; the news service parses it.
"""
from __future__ import annotations

from functools import lru_cache

from src.core.config import env
from src.core.logging import get_logger
from src.providers.base import provider_cfg

log = get_logger("providers.gemini")

_DISABLED = False
_FATAL = ("RESOURCE_EXHAUSTED", "PERMISSION_DENIED", "NOT_FOUND", "API_KEY_INVALID",
          "429", "403", "404")


def is_disabled() -> bool:
    return _DISABLED


def reset_breaker() -> None:  # for tests
    global _DISABLED
    _DISABLED = False


@lru_cache(maxsize=1)
def _client():
    e = env()
    if not e.gemini_api_key:
        log.warning("gemini_key_missing")
        return None
    try:
        from google import genai

        return genai.Client(api_key=e.gemini_api_key)
    except Exception as exc:  # pragma: no cover
        log.warning("gemini_init_failed", error=str(exc)[:160])
        return None


def available() -> bool:
    return (not _DISABLED) and _client() is not None


def generate_json(prompt: str) -> str | None:
    """Return raw JSON text from Gemini, or None (caller falls back to heuristic)."""
    global _DISABLED
    if _DISABLED:
        return None
    client = _client()
    if client is None:
        return None
    cfg = provider_cfg("news_scoring")
    model = cfg.get("model") or env().gemini_model
    try:
        resp = client.models.generate_content(
            model=model, contents=prompt,
            config={"response_mime_type": "application/json", "temperature": 0.0},
        )
        return resp.text
    except Exception as exc:
        msg = str(exc)
        if cfg.get("circuit_breaker", True) and any(mk in msg for mk in _FATAL):
            _DISABLED = True
            log.warning("gemini_breaker_tripped", reason=msg[:160])
        else:
            log.warning("gemini_call_failed", error=msg[:160])
        return None
