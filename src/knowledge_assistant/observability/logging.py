"""Structured logging configuration."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, MutableMapping
from typing import Any

import structlog

_SECRET_KEY = re.compile(
    r"(authorization|(?:api|event|signing)[_-]?key|secret|password|"
    r"(?:^|[_-])(?:access[_-]?|refresh[_-]?|auth[_-]?|bot[_-]?)?token$)",
    re.IGNORECASE,
)


def _redact_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: "[REDACTED]" if _SECRET_KEY.search(str(key)) else _redact_value(nested_value)
            for key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(item) for item in value)
    return value


def _redact_secrets(
    _logger: Any, _method_name: str, event_dict: MutableMapping[str, Any]
) -> Mapping[str, Any]:
    # `Any` is confined to this processor because structlog owns the event boundary.
    for key in list(event_dict):
        if _SECRET_KEY.search(key):
            event_dict[key] = "[REDACTED]"
        else:
            event_dict[key] = _redact_value(event_dict[key])
    return event_dict


def configure_logging(level: str) -> None:
    """Configure JSON logs suitable for local debugging and production ingestion."""

    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"unsupported log level: {level!r}")
    logging.basicConfig(level=numeric_level, format="%(message)s")

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _redact_secrets,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def bind_run_context(**values: str) -> None:
    structlog.contextvars.bind_contextvars(**values)


def clear_run_context() -> None:
    structlog.contextvars.clear_contextvars()
