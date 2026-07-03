"""Database engine + session management.

SQLite for the MVP; switch to PostgreSQL by changing ``DATABASE_URL`` only.
``init_db`` bootstraps the schema for local/dev use; Alembic owns migrations
for anything beyond a fresh dev database.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.core.config import env
from src.core.logging import get_logger
from src.db.models import Base

log = get_logger("db.session")

_url = env().database_url
_connect_args = {"check_same_thread": False} if _url.startswith("sqlite") else {}
engine = create_engine(_url, future=True, pool_pre_ping=True, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, class_=Session)


@contextmanager
def get_session() -> Iterator[Session]:
    """Transactional session scope: commits on success, rolls back on error."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Create all tables (dev bootstrap). Use Alembic for real migrations."""
    Base.metadata.create_all(engine)
    log.info("db_initialized", url=_url.split("://")[0], tables=len(Base.metadata.tables))
