"""Safe SQLite connection helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def open_read_only_database(path: Path) -> sqlite3.Connection:
    """Open the supplied knowledge database in immutable read-only mode."""

    resolved = path.resolve()
    connection = sqlite3.connect(
        f"file:{resolved.as_posix()}?mode=ro&immutable=1",
        uri=True,
        check_same_thread=False,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection
