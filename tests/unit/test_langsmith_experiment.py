import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
import pytest

from knowledge_assistant.evals.langsmith_experiment import (
    EXPECTED_EVALUATOR_KEYS,
    _new_evaluation_conversation_id,
    _sanitize_experiment_summary,
    _write_experiment_summary,
    require_complete_experiment_results,
    require_error_free_experiment,
)


def test_experiment_errors_fail_the_command() -> None:
    project = SimpleNamespace(error_rate=0.25)

    with pytest.raises(RuntimeError, match=r"error_rate=0\.25"):
        require_error_free_experiment(project, "experiment-name")


def test_error_free_experiment_is_accepted() -> None:
    require_error_free_experiment(SimpleNamespace(error_rate=0.0), "experiment-name")


def test_missing_experiment_error_rate_fails_the_command() -> None:
    with pytest.raises(RuntimeError, match="did not report an error rate"):
        require_error_free_experiment(SimpleNamespace(error_rate=None), "experiment-name")


def _completed_result_row(*, evaluator_keys: set[str] | None = None) -> dict[str, Any]:
    keys = EXPECTED_EVALUATOR_KEYS if evaluator_keys is None else evaluator_keys
    return {
        "run": SimpleNamespace(error=None),
        "evaluation_results": {
            "results": [SimpleNamespace(key=key) for key in keys],
        },
    }


def test_incomplete_experiment_result_count_fails_the_command() -> None:
    with pytest.raises(RuntimeError, match="returned 1 results; expected 2"):
        require_complete_experiment_results(
            [_completed_result_row()],
            expected_result_count=2,
            experiment_name="experiment-name",
        )


def test_missing_evaluator_feedback_fails_the_command() -> None:
    with pytest.raises(RuntimeError, match="missing evaluator keys"):
        require_complete_experiment_results(
            [_completed_result_row(evaluator_keys={"deterministic_pass"})],
            expected_result_count=1,
            experiment_name="experiment-name",
        )


def test_each_evaluation_repetition_gets_a_fresh_conversation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_ids = iter((uuid.UUID(int=1), uuid.UUID(int=2)))
    monkeypatch.setattr(
        "knowledge_assistant.evals.langsmith_experiment.uuid.uuid4",
        lambda: next(generated_ids),
    )

    first = _new_evaluation_conversation_id("balanced", "case-1")
    second = _new_evaluation_conversation_id("balanced", "case-1")

    assert first.startswith("langsmith:balanced:case-1:")
    assert second.startswith("langsmith:balanced:case-1:")
    assert first != second


async def test_experiment_summary_omits_private_langsmith_urls(tmp_path: Path) -> None:
    output_path = tmp_path / "summary.json"
    summary: dict[str, Any] = {
        "experiment_id": "experiment-id",
        "experiment_url": "private-experiment-link",
        "comparison_url": "private-comparison-link",
        "project_stats": {"error_rate": 0.0},
    }

    sanitized = _sanitize_experiment_summary(summary)
    await _write_experiment_summary(output_path, sanitized)
    persisted = json.loads(await anyio.Path(output_path).read_text(encoding="utf-8"))

    expected = {
        "experiment_id": "experiment-id",
        "project_stats": {"error_rate": 0.0},
    }
    assert sanitized == expected
    assert persisted == expected
