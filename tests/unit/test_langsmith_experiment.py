from types import SimpleNamespace

import pytest

from knowledge_assistant.evals.langsmith_experiment import require_error_free_experiment


def test_experiment_errors_fail_the_command() -> None:
    project = SimpleNamespace(error_rate=0.25)

    with pytest.raises(RuntimeError, match=r"error_rate=0\.25"):
        require_error_free_experiment(project, "experiment-name")


def test_error_free_experiment_is_accepted() -> None:
    require_error_free_experiment(SimpleNamespace(error_rate=0.0), "experiment-name")
