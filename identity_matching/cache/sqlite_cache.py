"""SQLite connections shared by the matching caches.

Only the connection layer lives here; each consumer owns its table schema.
"""

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

CACHE_DIR = Path(os.environ.get("MATCHING_CACHE_DIR", Path(__file__).resolve().parents[2] / "cache"))


def cache_db_path() -> Path:
    """Return the local, untracked SQLite database path."""
    return CACHE_DIR / "identity_matching_cache.sqlite"


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """Open a row-aware connection and commit or roll back before closing it."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    connection = sqlite3.connect(cache_db_path(), timeout=30)
    try:
        cache_db_path().chmod(0o600)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        with connection:
            yield connection
    finally:
        connection.close()
