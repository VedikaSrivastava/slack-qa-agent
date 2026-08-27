from __future__ import annotations

from typing import Any
from unittest.mock import patch

from knowledge_assistant.agent.processor import LangGraphQuestionProcessor, _create_chat_model
from knowledge_assistant.agent.profiles import (
    OPENAI_MAX_RETRIES,
    OPENAI_REQUEST_TIMEOUT_SECONDS,
    PRODUCTION_PROFILE,
)
from knowledge_assistant.agent.state import AgentState
from knowledge_assistant.config import AgentRuntimeSettings


class FakeGraph:
    def __init__(self) -> None:
        self.initial: AgentState | None = None
        self.config: dict[str, Any] | None = None

    async def ainvoke(self, initial: AgentState, *, config: dict[str, Any]) -> AgentState:
        self.initial = initial
        self.config = config
        return {
            "final_answer": (
                "I couldn't produce an answer that was fully supported by the knowledge base."
            ),
            "evidence": [
                {
                    "artifact_id": "A",
                    "title": "Artifact A",
                    "snippet": "Evidence",
                    "content": "Evidence",
                    "metadata": {},
                }
            ],
            "evidence_sufficient": True,
            "grounding_valid": False,
            "tool_call_count": 2,
            "model_call_count": 4,
            "retrieval_round_count": 1,
        }


async def test_rejected_ungrounded_answer_is_reported_as_insufficient() -> None:
    settings = AgentRuntimeSettings(
        _env_file=None,
        openai_api_key="test-key",
        database_url="postgresql+asyncpg://user:password@postgres/test",
    )
    graph = FakeGraph()
    processor = LangGraphQuestionProcessor(graph, settings, PRODUCTION_PROFILE)

    response = await processor.answer(
        question="What happened?",
        conversation_id="conversation",
        agent_run_id="run",
    )

    assert response.insufficient_evidence is True
    assert response.sources == []
    assert response.retrieved_artifact_ids == ["A"]
    assert response.model_call_count == 4
    assert graph.initial is not None
    assert graph.initial["final_answer"] == ""
    assert graph.initial["draft_answer"] == ""
    assert graph.initial["grounding_valid"] is False
    assert graph.config is not None
    assert graph.config["configurable"] == {"thread_id": "conversation"}


def test_chat_model_has_bounded_requests_and_no_nested_retries() -> None:
    settings = AgentRuntimeSettings(
        _env_file=None,
        openai_api_key="test-key",
        database_url="postgresql+asyncpg://user:password@postgres/test",
    )

    with patch("knowledge_assistant.agent.processor.ChatOpenAI") as chat_model:
        _create_chat_model(settings, PRODUCTION_PROFILE)

    chat_model.assert_called_once_with(
        api_key=settings.openai_api_key,
        model=PRODUCTION_PROFILE.model_name,
        max_retries=OPENAI_MAX_RETRIES,
        timeout=OPENAI_REQUEST_TIMEOUT_SECONDS,
        temperature=PRODUCTION_PROFILE.temperature,
    )
