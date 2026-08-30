from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from knowledge_assistant.agent.models import (
    FinalAnswerEvent,
    ProgressEvent,
    ProgressStage,
    QuestionDisposition,
)
from knowledge_assistant.agent.processor import (
    AgentGraph,
    LangGraphQuestionProcessor,
    _create_langfuse_handler,
)
from knowledge_assistant.agent.profiles import PRODUCTION_PROFILE
from knowledge_assistant.agent.retrieval_tools import KnowledgeRetrievalTools
from knowledge_assistant.agent.state import AgentState
from knowledge_assistant.agent.workflow_nodes import GroundedAnswerNodes
from knowledge_assistant.config import AgentRuntimeSettings


@dataclass(frozen=True)
class FakeSnapshot:
    values: Mapping[str, Any]
    next: tuple[str, ...]


class FakeGraph:
    def __init__(
        self,
        *,
        checkpoint_state: AgentState | None = None,
        next_nodes: tuple[str, ...] = (),
        updates: list[tuple[str, AgentState]] | None = None,
    ) -> None:
        self.state = cast(AgentState, dict(checkpoint_state or {}))
        self.next_nodes = next_nodes
        self.updates = updates or []
        self.config: dict[str, Any] | None = None
        self.stream_input: AgentState | None = None
        self.stream_calls = 0
        self.stream_mode: str | None = None
        self.stream_version: str | None = None

    async def aget_state(self, config: dict[str, Any]) -> FakeSnapshot:
        self.config = config
        return FakeSnapshot(values=dict(self.state), next=self.next_nodes)

    async def astream(
        self,
        input_state: AgentState | None,
        *,
        config: dict[str, Any],
        stream_mode: Literal["updates"],
        version: Literal["v2"],
    ) -> AsyncIterator[dict[str, Any]]:
        self.stream_calls += 1
        self.stream_input = input_state
        self.config = config
        self.stream_mode = stream_mode
        self.stream_version = version
        if input_state is not None:
            self.state.update(input_state)
        for node_name, update in self.updates:
            self.state.update(update)
            yield {"type": "updates", "ns": (), "data": {node_name: update}}
        self.next_nodes = ()


def _settings() -> AgentRuntimeSettings:
    return AgentRuntimeSettings(
        _env_file=None,
        openai_api_key="test-key",
        database_url="postgresql+asyncpg://user:password@postgres/test",
    )


@patch("knowledge_assistant.agent.processor.CallbackHandler")
@patch("knowledge_assistant.agent.processor.Langfuse")
def test_langfuse_handler_uses_parsed_settings(
    langfuse_client: Mock,
    callback_handler: Mock,
) -> None:
    settings = AgentRuntimeSettings(
        _env_file=None,
        app_env="test",
        openai_api_key="test-key",
        database_url="postgresql+asyncpg://user:password@postgres/test",
        langfuse_base_url="http://langfuse.test:3000",
        langfuse_public_key="lf_pk_test",
        langfuse_secret_key="lf_sk_test",
    )

    handler = _create_langfuse_handler(settings)

    langfuse_client.assert_called_once_with(
        public_key="lf_pk_test",
        secret_key="lf_sk_test",
        # No trailing slash: the SDK appends the OTEL ingestion path to this value, and a
        # double slash makes Langfuse redirect the export away with a 308.
        base_url="http://langfuse.test:3000",
        environment="test",
        release="0.1.0",
    )
    callback_handler.assert_called_once_with(public_key="lf_pk_test")
    assert handler is callback_handler.return_value


@patch("knowledge_assistant.agent.processor.CallbackHandler")
@patch("knowledge_assistant.agent.processor.Langfuse")
def test_langfuse_handler_is_absent_without_credentials(
    langfuse_client: Mock,
    callback_handler: Mock,
) -> None:
    assert _create_langfuse_handler(_settings()) is None
    langfuse_client.assert_not_called()
    callback_handler.assert_not_called()


def _evidence() -> dict[str, Any]:
    return {
        "artifact_id": "art_A",
        "title": "Artifact A",
        "snippet": "Evidence",
        "content": "Evidence",
        "metadata": {},
    }


def _completed_state(*, agent_run_id: str = "run") -> AgentState:
    return {
        "question": "What happened?",
        "standalone_question": "What happened?",
        "agent_run_id": agent_run_id,
        "conversation_id": "conversation",
        "question_disposition": QuestionDisposition.KNOWLEDGE_QUESTION,
        "final_answer": "Supported answer [art_A]",
        "evidence": [_evidence()],
        "evidence_sufficient": True,
        "grounding_valid": True,
        "tool_call_count": 2,
        "model_call_count": 4,
        "retrieval_round_count": 1,
        "input_tokens": 120,
        "output_tokens": 30,
    }


async def test_rejected_ungrounded_answer_is_reported_as_insufficient() -> None:
    graph = FakeGraph(
        updates=[
            (
                "finalize",
                {
                    "final_answer": (
                        "I couldn't produce an answer that was fully supported by the "
                        "knowledge base."
                    ),
                    "evidence": [_evidence()],
                    "evidence_sufficient": True,
                    "grounding_valid": False,
                    "is_abstention": True,
                    "tool_call_count": 2,
                    "model_call_count": 4,
                    "retrieval_round_count": 1,
                },
            )
        ]
    )
    processor = LangGraphQuestionProcessor(graph, _settings(), PRODUCTION_PROFILE)

    response = await processor.answer(
        question="What happened?",
        conversation_id="conversation",
        agent_run_id="run",
    )

    assert response.insufficient_evidence is True
    assert response.sources == []
    assert response.retrieved_artifact_ids == ["art_A"]
    assert response.model_call_count == 4
    assert graph.stream_input is not None
    assert graph.stream_input["final_answer"] == ""
    assert graph.stream_input["draft_answer"] == ""
    assert graph.stream_input["grounding_valid"] is False
    assert graph.stream_input["input_tokens"] == 0
    assert graph.stream_input["output_tokens"] == 0
    assert graph.config is not None
    assert graph.config["configurable"] == {"thread_id": "conversation"}
    assert graph.stream_mode == "updates"
    assert graph.stream_version == "v2"


async def test_internal_grader_uncertainty_does_not_mislabel_a_substantive_answer() -> None:
    graph = FakeGraph(
        updates=[
            (
                "finalize",
                {
                    "final_answer": "The supported result is 12 accounts [art_A].",
                    "evidence": [_evidence()],
                    "evidence_sufficient": False,
                    "grounding_valid": False,
                    "is_abstention": False,
                },
            )
        ]
    )
    processor = LangGraphQuestionProcessor(graph, _settings(), PRODUCTION_PROFILE)

    response = await processor.answer(
        question="How many accounts are there?",
        conversation_id="conversation",
        agent_run_id="run",
    )

    assert response.insufficient_evidence is False


async def test_completed_same_run_reconstructs_response_without_rerunning_graph() -> None:
    graph = FakeGraph(checkpoint_state=_completed_state())
    processor = LangGraphQuestionProcessor(graph, _settings(), PRODUCTION_PROFILE)

    events = [
        event
        async for event in processor.run(
            question="What happened?",
            conversation_id="conversation",
            agent_run_id="run",
        )
    ]

    assert len(events) == 1
    final_event = events[0]
    assert isinstance(final_event, FinalAnswerEvent)
    assert final_event.event_id == "run:final"
    assert final_event.response.answer == "Supported answer [art_A]"
    assert final_event.response.input_tokens == 120
    assert final_event.response.output_tokens == 30
    assert [source.artifact_id for source in final_event.response.sources] == ["art_A"]
    assert graph.stream_calls == 0


async def test_processor_streams_and_recovers_with_a_real_langgraph_checkpoint() -> None:
    node_calls = 0

    async def finalize_run(state: AgentState) -> AgentState:
        nonlocal node_calls
        node_calls += 1
        return {
            "final_answer": f"Answer for {state['question']}",
            "evidence": [],
            "evidence_sufficient": False,
            "grounding_valid": True,
        }

    builder = StateGraph(AgentState)
    builder.add_node("finalize", cast(Any, finalize_run))
    builder.add_edge(START, "finalize")
    builder.add_edge("finalize", END)
    graph = cast(AgentGraph, builder.compile(checkpointer=InMemorySaver()))
    processor = LangGraphQuestionProcessor(graph, _settings(), PRODUCTION_PROFILE)

    first_events = [
        event
        async for event in processor.run(
            question="What happened?",
            conversation_id="conversation",
            agent_run_id="run",
        )
    ]
    recovered_events = [
        event
        async for event in processor.run(
            question="What happened?",
            conversation_id="conversation",
            agent_run_id="run",
        )
    ]

    assert [event.kind for event in first_events] == ["progress", "final_answer"]
    assert [event.kind for event in recovered_events] == ["final_answer"]
    assert isinstance(recovered_events[0], FinalAnswerEvent)
    assert recovered_events[0].response.answer == "Answer for What happened?"
    assert node_calls == 1


async def test_incomplete_same_run_resumes_without_new_input_or_private_progress_text() -> None:
    checkpoint_state = _completed_state()
    checkpoint_state.update(
        {
            "final_answer": "",
            "draft_answer": "PRIVATE DRAFT MUST NOT LEAK [art_A]",
            "grounding_valid": False,
        }
    )
    graph = FakeGraph(
        checkpoint_state=checkpoint_state,
        next_nodes=("verify_grounding",),
        updates=[
            ("verify_grounding", {"grounding_valid": True, "grounding_issues": []}),
            (
                "finalize",
                {
                    "final_answer": "Supported answer [art_A]",
                    "history": [
                        {
                            "agent_run_id": "run",
                            "question": "What happened?",
                            "answer": "Supported answer [art_A]",
                        }
                    ],
                },
            ),
        ],
    )
    processor = LangGraphQuestionProcessor(graph, _settings(), PRODUCTION_PROFILE)

    events = [
        event
        async for event in processor.run(
            question="What happened?",
            conversation_id="conversation",
            agent_run_id="run",
        )
    ]

    assert graph.stream_input is None
    first_event = events[0]
    assert isinstance(first_event, ProgressEvent)
    assert first_event.stage is ProgressStage.VERIFYING
    assert first_event.sequence == 70
    progress_json = " ".join(
        event.model_dump_json() for event in events if isinstance(event, ProgressEvent)
    )
    assert "PRIVATE DRAFT" not in progress_json
    assert isinstance(events[-1], FinalAnswerEvent)


async def test_different_run_starts_a_new_turn_without_overwriting_conversation_thread() -> None:
    old_state = _completed_state(agent_run_id="old-run")
    old_state["history"] = [
        {
            "agent_run_id": "old-run",
            "question": "Earlier question",
            "answer": "Earlier answer",
        }
    ]
    graph = FakeGraph(
        checkpoint_state=old_state,
        updates=[
            (
                "finalize",
                {
                    "final_answer": "New answer",
                    "evidence": [],
                    "evidence_sufficient": False,
                    "grounding_valid": True,
                },
            )
        ],
    )
    processor = LangGraphQuestionProcessor(graph, _settings(), PRODUCTION_PROFILE)

    await processor.answer(
        question="New question",
        conversation_id="conversation",
        agent_run_id="new-run",
    )

    assert graph.stream_input is not None
    assert graph.stream_input["agent_run_id"] == "new-run"
    assert graph.stream_input["question"] == "New question"
    assert "history" not in graph.stream_input
    assert graph.config is not None
    assert graph.config["configurable"] == {"thread_id": "conversation"}


async def test_progress_is_stable_and_never_contains_graph_payloads() -> None:
    private_text = "PRIVATE MODEL AUDIT DETAIL"
    graph = FakeGraph(
        updates=[
            ("plan_retrieval", {"search_queries": [private_text]}),
            (
                "execute_retrieval",
                {"evidence": [_evidence()], "retrieval_round_count": 1},
            ),
            (
                "grade_evidence",
                {"evidence_sufficient": True, "insufficiency_reason": private_text},
            ),
            ("generate_answer", {"draft_answer": private_text}),
            (
                "verify_grounding",
                {"grounding_valid": False, "grounding_issues": [private_text]},
            ),
            ("repair_answer", {"draft_answer": "Supported answer [art_A]"}),
            ("verify_repair", {"grounding_valid": True, "grounding_issues": []}),
            ("finalize", {"final_answer": "Supported answer [art_A]"}),
        ]
    )
    processor = LangGraphQuestionProcessor(graph, _settings(), PRODUCTION_PROFILE)

    events = [
        event
        async for event in processor.run(
            question="What happened?",
            conversation_id="conversation",
            agent_run_id="run",
        )
    ]

    progress_events = [event for event in events if isinstance(event, ProgressEvent)]
    assert [event.sequence for event in progress_events] == [10, 20, 30, 60, 70, 80, 90]
    assert [event.stage for event in progress_events] == [
        ProgressStage.THINKING,
        ProgressStage.SEARCHING,
        ProgressStage.REVIEWING,
        ProgressStage.DRAFTING,
        ProgressStage.VERIFYING,
        ProgressStage.TIGHTENING,
        ProgressStage.VERIFYING,
    ]
    assert len({event.event_id for event in progress_events}) == len(progress_events)
    assert private_text not in " ".join(event.model_dump_json() for event in progress_events)
    assert isinstance(events[-1], FinalAnswerEvent)
    assert events[-1].event_id == "run:final"


async def test_finalize_replaces_history_turn_for_same_run_id() -> None:
    nodes = GroundedAnswerNodes(
        cast(BaseChatModel, object()),
        cast(KnowledgeRetrievalTools, object()),
        PRODUCTION_PROFILE,
    )
    state: AgentState = {
        "agent_run_id": "run",
        "question": "Question",
        "draft_answer": "Final answer",
        "history": [
            {"agent_run_id": "older", "question": "Old", "answer": "Old answer"},
            {"agent_run_id": "run", "question": "Question", "answer": "Stale answer"},
        ],
    }

    first_result = await nodes.finalize(state)
    second_result = await nodes.finalize({**state, "history": first_result["history"]})

    assert second_result["history"] == [
        {"agent_run_id": "older", "question": "Old", "answer": "Old answer"},
        {
            "agent_run_id": "run",
            "question": "Question",
            "answer": "Final answer",
            "sources": [],
            "retrieved_artifact_ids": [],
        },
    ]


async def test_model_usage_is_accumulated_into_checkpointed_state() -> None:
    model = Mock()
    model.ainvoke = AsyncMock(
        return_value=AIMessage(
            content="Supported answer [art_A]",
            usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
        )
    )
    nodes = GroundedAnswerNodes(
        cast(BaseChatModel, model),
        cast(KnowledgeRetrievalTools, object()),
        PRODUCTION_PROFILE,
    )
    state: AgentState = {
        "standalone_question": "Question",
        "evidence": [_evidence()],
        "model_call_count": 1,
        "input_tokens": 7,
        "output_tokens": 3,
    }

    result = await nodes.generate_answer(state)

    assert result["input_tokens"] == 17
    assert result["output_tokens"] == 7
    assert result["model_call_count"] == 2
