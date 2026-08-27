import pytest

from knowledge_assistant.observability.logging import _redact_secrets, configure_logging


def test_secret_redaction_traverses_nested_mappings_and_sequences() -> None:
    event = {
        "event": "request_failed",
        "headers": {"Authorization": "Bearer secret", "content_type": "application/json"},
        "attempts": [{"api_key": "secret", "status": 401}],
        "token": "top-level-secret",
        "slack_bot_token": "bot-secret",
        "input_tokens": 123,
        "token_count": 456,
    }

    redacted = _redact_secrets(None, "error", event)

    assert redacted == {
        "event": "request_failed",
        "headers": {"Authorization": "[REDACTED]", "content_type": "application/json"},
        "attempts": [{"api_key": "[REDACTED]", "status": 401}],
        "token": "[REDACTED]",
        "slack_bot_token": "[REDACTED]",
        "input_tokens": 123,
        "token_count": 456,
    }


def test_logging_configuration_rejects_unknown_level() -> None:
    with pytest.raises(ValueError, match="unsupported log level"):
        configure_logging("VERBOSE")
