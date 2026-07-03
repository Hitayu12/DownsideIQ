"""Celery application (async job queue, Redis broker/backend).

The API enqueues long-running work (analyze/train/backtest) here and polls for
results. For local dev without Redis, set ``CELERY_EAGER=true`` to run tasks
inline (synchronously) in-process.
"""
from __future__ import annotations

import os

from celery import Celery

from src.core.config import env

_eager = os.getenv("CELERY_EAGER", "false").lower() in ("1", "true", "yes")
# Eager (dev/test) needs no Redis: run inline with an in-process result backend
# so the enqueue->poll pattern works without external infra.
_broker = "memory://" if _eager else env().redis_url
_backend = "cache+memory://" if _eager else env().redis_url

celery_app = Celery("downsideiq", broker=_broker, backend=_backend, include=["src.tasks.jobs"])
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_time_limit=600,
    task_soft_time_limit=540,
    task_always_eager=_eager,
    task_eager_propagates=_eager,
    task_store_eager_result=_eager,   # persist eager results so /jobs polling works
    result_expires=3600,
)
