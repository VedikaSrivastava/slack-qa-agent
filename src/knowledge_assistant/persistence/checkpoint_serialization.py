"""Restricted LangGraph checkpoint serialization for application state."""

from __future__ import annotations

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from knowledge_assistant.agent.models import QuestionDisposition


def create_checkpoint_serializer() -> JsonPlusSerializer:
    """Allow only the custom type intentionally persisted in ``AgentState``.

    LangGraph already permits its built-in safe types. Keeping this application allowlist explicit
    prevents a compromised checkpoint store from importing arbitrary Python types during restore.
    """

    return JsonPlusSerializer(allowed_msgpack_modules=(QuestionDisposition,))
