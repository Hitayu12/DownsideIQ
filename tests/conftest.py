"""Pytest session setup: run Celery inline (no Redis) and isolate the test DB."""
from __future__ import annotations

import os

# Must be set before any src.tasks.celery_app import so eager mode + memory
# backend take effect (no external Redis needed in tests).
os.environ.setdefault("CELERY_EAGER", "true")
# Use a throwaway SQLite DB for tests so we never touch the dev/prod database.
os.environ.setdefault("DATABASE_URL", "sqlite:///test_downsideiq.db")
