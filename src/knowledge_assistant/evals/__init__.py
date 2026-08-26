"""Deterministic and optional model-backed evaluation infrastructure."""

from knowledge_assistant.evals.runner import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
