"""Shared provider plumbing: config access + retry/timeout wrapper.

Every external call goes through ``call_with_retry`` so retries, backoff, and
structured failure logging are uniform. Transient failures are retried with
exponential backoff; exhaustion raises ``ProviderError`` (caller decides whether
that is fatal — prices — or degradable — news/fundamentals).
"""
from __future__ import annotations

from typing import Any, Callable, TypeVar

from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.core.config import get_data_sources
from src.core.errors import ProviderError
from src.core.logging import get_logger

log = get_logger("providers")
T = TypeVar("T")


def provider_cfg(section: str) -> dict[str, Any]:
    return get_data_sources().get(section, {})


def call_with_retry(
    fn: Callable[[], T],
    *,
    provider: str,
    retries: int = 2,
    backoff: float = 2.0,
    transient: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """Run ``fn`` with exponential-backoff retries; raise ProviderError on exhaustion."""
    try:
        for attempt in Retrying(
            stop=stop_after_attempt(retries + 1),
            wait=wait_exponential(multiplier=backoff, max=30),
            retry=retry_if_exception_type(transient),
            reraise=True,
        ):
            with attempt:
                n = attempt.retry_state.attempt_number
                if n > 1:
                    log.warning("provider_retry", provider=provider, attempt=n)
                return fn()
    except RetryError as exc:  # pragma: no cover - reraise=True normally re-raises inner
        raise ProviderError(provider, f"retries exhausted: {exc}", transient=True)
    except Exception as exc:
        raise ProviderError(provider, str(exc)[:200], transient=True) from exc
    raise ProviderError(provider, "unreachable")  # pragma: no cover
