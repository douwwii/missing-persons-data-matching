import pytest

from identity_matching.cache import sqlite_cache
from identity_matching.matching import blocking


@pytest.fixture(autouse=True)
def isolate_matching_caches(monkeypatch, tmp_path):
    """Keep test records out of the application's operational caches."""
    monkeypatch.setattr(sqlite_cache, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(blocking, "BLOCKING_INDEX_CACHE_DIR", tmp_path / "blocking_indexes")
    blocking._BLOCKING_INDEX_CACHE.clear()
    yield
    blocking._BLOCKING_INDEX_CACHE.clear()
