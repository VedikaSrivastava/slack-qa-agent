"""Safe SQLite connection helpers."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def open_read_only_database(path: Path) -> Iterator[sqlite3.Connection]:
    """Yield an immutable read-only connection and always close it afterward."""

    resolved = path.resolve()
    connection = sqlite3.connect(
        f"{resolved.as_uri()}?mode=ro&immutable=1",
        uri=True,
    )
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        yield connection
    finally:
        connection.close()
