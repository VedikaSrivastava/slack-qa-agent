# Experiment summaries

LangSmith experiment commands write one JSON summary per code-defined agent profile here. These
small summaries are safe to review and commit; credentials, raw environment files, and the supplied
SQLite database are never written to this directory.

The LangSmith experiment itself remains the canonical record for per-example traces and evaluator
feedback. A summary records its immutable gold-dataset digest and version tag, profile parameters,
latency, token, cost, error, and evaluator statistics so a production-profile change can be reviewed
in Git. Private LangSmith experiment and comparison URLs are deliberately omitted from both command
output and these files.
