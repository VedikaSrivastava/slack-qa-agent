import json

from structlog.testing import capture_logs

from knowledge_assistant.execution.inngest import _log_safe_error_publish_failure
from knowledge_assistant.execution.models import QuestionJob


def test_safe_error_publish_log_omits_exception_text() -> None:
    secret_value = "provider-detail-that-must-not-be-logged"
    job = QuestionJob(
        event_id="Ev1",
        team_id="T1",
        channel_id="C1",
        user_id="U1",
        message_ts="2.0",
        thread_ts="1.0",
        question="What changed?",
    )

    with capture_logs() as captured_logs:
        _log_safe_error_publish_failure(job, RuntimeError(secret_value))

    assert len(captured_logs) == 1
    event = captured_logs[0]
    assert event["event"] == "slack_safe_error_publish_failed"
    assert event["error_code"] == "slack_safe_error_publish_failed"
    assert event["exception_class"] == "RuntimeError"
    assert secret_value not in json.dumps(captured_logs)
    assert "exception" not in event
    assert "exc_info" not in event
