"""Structured logging (structlog over stdlib).

Emits JSON logs (console + rotating file) while remaining compatible with the
ported MVP code that uses ``log.info("msg %s", value)`` printf style (handled by
``PositionalArgumentsFormatter``). Use ``get_logger(name)`` everywhere.
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

import structlog

_CONFIGURED = False


def _shared_processors():
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    from src.core.config import env, project_root

    e = env()
    level = getattr(logging, e.log_level.upper(), logging.INFO)
    log_dir = project_root() / e.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    shared = _shared_processors()
    structlog.configure(
        processors=shared + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    renderer = (
        structlog.processors.JSONRenderer() if e.log_json
        else structlog.dev.ConsoleRenderer(colors=True)
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
    )

    root = logging.getLogger("downsideiq")
    root.setLevel(level)
    root.handlers.clear()
    root.propagate = False

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    fileh = RotatingFileHandler(log_dir / "downsideiq.jsonl", maxBytes=5_000_000,
                                backupCount=5, encoding="utf-8")
    fileh.setFormatter(formatter)
    root.addHandler(fileh)

    _CONFIGURED = True


def get_logger(name: str):
    """Return a structlog logger namespaced under ``downsideiq``."""
    configure_logging()
    return structlog.get_logger(f"downsideiq.{name}")
