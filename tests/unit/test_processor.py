from __future__ import annotations

from typing import Any

from knowledge_assistant.agent.processor import LangGraphQuestionProcessor
from knowledge_assistant.agent.profiles import PRODUCTION_PROFILE
from knowledge_assistant.config import AgentRuntimeSettings


class FakeGraph:
    def __init__(self) -> None:
        self.initial: dict[str, Any] | None = None

    async def ainvoke(self, initial: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
        del config
        self.initial = initial
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
    assert graph.initial is not None
    assert graph.initial["final_answer"] == ""
    assert graph.initial["draft_answer"] == ""
    assert graph.initial["grounding_valid"] is False
