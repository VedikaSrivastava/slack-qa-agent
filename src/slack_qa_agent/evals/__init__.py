"""Deterministic and optional model-backed evaluation infrastructure."""

from slack_qa_agent.evals.runner import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
