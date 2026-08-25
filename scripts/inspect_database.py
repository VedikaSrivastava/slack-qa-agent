"""Inspect the supplied SQLite database before designing retrieval."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

DEFAULT_PATH = Path("data/synthetic_startup.sqlite")


def inspect_database(path: Path) -> None:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Database not found at {resolved}. Download synthetic_startup.sqlite first."
        )

    connection = sqlite3.connect(
        f"file:{resolved.as_posix()}?mode=ro&immutable=1",
        uri=True,
    )
    try:
        tables = connection.execute(
            """
            SELECT name, sql
            FROM sqlite_master
            WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()

        print(f"Database: {resolved}")
        print(f"Objects: {len(tables)}\n")
        for name, sql in tables:
            quoted_name = '"' + str(name).replace('"', '""') + '"'
            count = connection.execute(f"SELECT COUNT(*) FROM {quoted_name}").fetchone()[0]
            print(f"[{name}] rows={count}")
            print(sql or "<no schema>")
            print()
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    inspect_database(args.path)


if __name__ == "__main__":
    main()
