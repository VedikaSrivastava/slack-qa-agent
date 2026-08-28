# Synthetic Startup Take-Home Database

The supplied dataset is already included; no download or extraction step is required.

## Files

- `synthetic_startup.sqlite`: extracted SQLite database read by the application.
- `synthetic_startup_takehome_db.zip`: original delivery archive, retained for provenance.

The original delivery README remains inside the ZIP. SQLite-generated `-wal` and `-shm` sidecars
are runtime state and are intentionally ignored.

The application opens `synthetic_startup.sqlite` in immutable read-only mode. To inspect its schema
and row counts through the same connection boundary used by runtime retrieval, run:

```bash
uv run python scripts/inspect_database.py
```

The main long-form corpus lives in `artifacts`; `artifacts_fts` is its full-text search index.
