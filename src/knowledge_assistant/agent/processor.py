"""LangGraph-backed implementation of the application QuestionProcessor."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any, Literal, Protocol, cast

import anyio
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from knowledge_assistant.agent.citations import cited_artifact_ids
from knowledge_assistant.agent.graph import build_graph
from knowledge_assistant.agent.models import (
    AgentResponse,
    EvidenceReference,
    FinalAnswerEvent,
    ProcessorEvent,
    ProgressEvent,
    ProgressStage,
    QuestionDisposition,
)
from knowledge_assistant.agent.profiles import (
    OPENAI_MAX_RETRIES,
    OPENAI_REQUEST_TIMEOUT_SECONDS,
    AgentProfile,
)
from knowledge_assistant.agent.retrieval_tools import KnowledgeRetrievalTools
from knowledge_assistant.agent.state import AgentState
from knowledge_assistant.agent.workflow_nodes import GroundedAnswerNodes
from knowledge_assistant.application.question_processor import StreamingQuestionProcessor
from knowledge_assistant.config import (
    APPLICATION_VERSION,
    PROMPT_VERSION,
    RETRIEVAL_VERSION,
    AgentRuntimeSettings,
)
from knowledge_assistant.retrieval.models import EvidenceItem
from knowledge_assistant.retrieval.repository import SQLiteKnowledgeRepository


class AgentStateSnapshot(Protocol):
    """Checkpoint fields needed to distinguish new, resumable, and completed runs."""

    @property
    def values(self) -> Mapping[str, Any]: ...

    @property
    def next(self) -> tuple[str, ...]: ...


class AgentGraph(Protocol):
    """Narrow portion of a checkpointed compiled graph used by the processor."""

    async def aget_state(self, config: dict[str, Any]) -> AgentStateSnapshot: ...

    def astream(
        self,
        state: AgentState | None,
        *,
        config: dict[str, Any],
        stream_mode: Literal["updates"],
        version: Literal["v2"],
    ) -> AsyncIterator[dict[str, Any]]: ...


class LangGraphQuestionProcessor:
    def __init__(
        self,
        graph: AgentGraph,
        settings: AgentRuntimeSettings,
        profile: AgentProfile,
    ) -> None:
        self._graph = graph
        self._settings = settings
        self._profile = profile

    async def answer(
        self,
        *,
        question: str,
        conversation_id: str,
        agent_run_id: str,
    ) -> AgentResponse:
        response: AgentResponse | None = None
        async for event in self.run(
            question=question,
            conversation_id=conversation_id,
            agent_run_id=agent_run_id,
        ):
            if not isinstance(event, FinalAnswerEvent):
                continue
            if response is not None:
                raise RuntimeError("Processor emitted more than one final answer")
            response = event.response
        if response is None:
            raise RuntimeError("Processor stream ended without a final answer")
        return response

    async def run(
        self,
        *,
        question: str,
        conversation_id: str,
        agent_run_id: str,
    ) -> AsyncIterator[ProcessorEvent]:
        """Run or resume one durable graph invocation and expose only sanitized events."""

        config = self._graph_config(
            conversation_id=conversation_id,
            agent_run_id=agent_run_id,
        )
        snapshot = await self._graph.aget_state(config)
        checkpoint_state = cast(AgentState, dict(snapshot.values))
        next_nodes = tuple(snapshot.next)
        is_same_run = checkpoint_state.get("agent_run_id") == agent_run_id

        if is_same_run and not next_nodes and checkpoint_state.get("final_answer"):
            yield self._final_event(agent_run_id, checkpoint_state)
            return

        emitted_sequences: set[int] = set()
        if is_same_run and next_nodes:
            stream_input: AgentState | None = None
            accumulated_state = checkpoint_state
            for node_name in next_nodes:
                progress = self._progress_before_node(node_name, accumulated_state)
                if progress is not None:
                    event = self._progress_event(agent_run_id, progress)
                    if event.sequence not in emitted_sequences:
                        emitted_sequences.add(event.sequence)
                        yield event
        else:
            # LangGraph merges this input with the conversation checkpoint. Reset every current-run
            # field, but intentionally omit `history` so prior turns in the same thread survive.
            stream_input = self._initial_state(
                question=question,
                conversation_id=conversation_id,
                agent_run_id=agent_run_id,
            )
            accumulated_state = cast(AgentState, dict(stream_input))
            event = self._progress_event(
                agent_run_id,
                (10, ProgressStage.THINKING, None),
            )
            emitted_sequences.add(event.sequence)
            yield event

        async for part in self._graph.astream(
            stream_input,
            config=config,
            stream_mode="updates",
            version="v2",
        ):
            for node_name, update in _stream_updates(part):
                accumulated_state.update(update)
                progress = self._progress_after_node(node_name, accumulated_state)
                if progress is None:
                    continue
                event = self._progress_event(agent_run_id, progress)
                if event.sequence in emitted_sequences:
                    continue
                emitted_sequences.add(event.sequence)
                yield event

        completed_snapshot = await self._graph.aget_state(config)
        completed_state = cast(AgentState, dict(completed_snapshot.values))
        if (
            completed_state.get("agent_run_id") != agent_run_id
            or completed_snapshot.next
            or not completed_state.get("final_answer")
        ):
            raise RuntimeError("Graph stream ended before this run reached a final checkpoint")
        yield self._final_event(agent_run_id, completed_state)

    def _initial_state(
        self,
        *,
        question: str,
        conversation_id: str,
        agent_run_id: str,
    ) -> AgentState:
        return {
            "question": question,
            "standalone_question": question,
            "agent_run_id": agent_run_id,
            "conversation_id": conversation_id,
            "question_disposition": QuestionDisposition.KNOWLEDGE_QUESTION,
            "search_queries": [],
            "account_lookup": None,
            "evidence": [],
            "retrieval_round_count": 0,
            "tool_call_count": 0,
            "model_call_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "evidence_sufficient": False,
            "insufficiency_reason": "",
            "draft_answer": "",
            "final_answer": "",
            "grounding_valid": False,
            "grounding_issues": [],
        }

    def _graph_config(
        self,
        *,
        conversation_id: str,
        agent_run_id: str,
    ) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": conversation_id},
            "recursion_limit": 20,
            "metadata": {
                "agent_run_id": agent_run_id,
                "conversation_id": conversation_id,
                "prompt_version": PROMPT_VERSION,
                "retrieval_version": RETRIEVAL_VERSION,
                "model": self._profile.model_name,
                "agent_profile": self._profile.name,
                "environment": self._settings.app_env,
                "application_version": APPLICATION_VERSION,
            },
        }

    def _final_event(self, agent_run_id: str, state: AgentState) -> FinalAnswerEvent:
        return FinalAnswerEvent(
            agent_run_id=agent_run_id,
            event_id=f"{agent_run_id}:final",
            response=_response_from_state(state),
        )

    def _progress_event(
        self,
        agent_run_id: str,
        progress: tuple[int, ProgressStage, int | None],
    ) -> ProgressEvent:
        sequence, stage, retrieval_round = progress
        return ProgressEvent(
            agent_run_id=agent_run_id,
            event_id=f"{agent_run_id}:{sequence}",
            sequence=sequence,
            stage=stage,
            retrieval_round=retrieval_round,
        )

    def _progress_before_node(
        self,
        node_name: str,
        state: AgentState,
    ) -> tuple[int, ProgressStage, int | None] | None:
        retrieval_round = _bounded_retrieval_round(state.get("retrieval_round_count", 0))
        if node_name in {"resolve_question", "plan_retrieval"}:
            return 10, ProgressStage.THINKING, None
        if node_name == "execute_retrieval":
            if retrieval_round == 0:
                return 20, ProgressStage.SEARCHING, 1
            return 40, ProgressStage.SEARCHING, 2
        if node_name == "grade_evidence":
            if retrieval_round <= 1:
                return 30, ProgressStage.REVIEWING, 1
            return 50, ProgressStage.REVIEWING, 2
        if node_name == "refine_retrieval":
            return 40, ProgressStage.SEARCHING, 2
        if node_name == "generate_answer":
            return 60, ProgressStage.DRAFTING, None
        if node_name == "verify_grounding":
            return 70, ProgressStage.VERIFYING, None
        if node_name == "repair_answer":
            return 80, ProgressStage.TIGHTENING, None
        if node_name in {"verify_repair", "reject_ungrounded_answer"}:
            return 90, ProgressStage.VERIFYING, None
        return None

    def _progress_after_node(
        self,
        node_name: str,
        state: AgentState,
    ) -> tuple[int, ProgressStage, int | None] | None:
        retrieval_round = _bounded_retrieval_round(state.get("retrieval_round_count", 0))
        if node_name == "plan_retrieval":
            if (
                QuestionDisposition(state["question_disposition"])
                is not QuestionDisposition.KNOWLEDGE_QUESTION
            ):
                return None
            return 20, ProgressStage.SEARCHING, 1
        if node_name == "execute_retrieval":
            if retrieval_round <= 1:
                return 30, ProgressStage.REVIEWING, 1
            return 50, ProgressStage.REVIEWING, 2
        if node_name == "grade_evidence":
            if state.get("evidence_sufficient") or (
                retrieval_round >= self._profile.max_retrieval_rounds
            ):
                return 60, ProgressStage.DRAFTING, None
            return 40, ProgressStage.SEARCHING, 2
        if node_name == "refine_retrieval":
            return 40, ProgressStage.SEARCHING, 2
        if node_name == "generate_answer":
            return 70, ProgressStage.VERIFYING, None
        if node_name == "verify_grounding" and not state.get("grounding_valid"):
            return 80, ProgressStage.TIGHTENING, None
        if node_name == "repair_answer":
            return 90, ProgressStage.VERIFYING, None
        return None


def _stream_updates(part: object) -> list[tuple[str, AgentState]]:
    """Extract graph state updates internally without exposing their untrusted payloads."""

    if not isinstance(part, Mapping) or part.get("type") != "updates":
        return []
    raw_data = part.get("data")
    if not isinstance(raw_data, Mapping):
        return []
    updates: list[tuple[str, AgentState]] = []
    for node_name, raw_update in raw_data.items():
        if isinstance(node_name, str) and isinstance(raw_update, Mapping):
            updates.append((node_name, cast(AgentState, dict(raw_update))))
    return updates


def _bounded_retrieval_round(value: object) -> int:
    if not isinstance(value, int) or value <= 0:
        return 0
    return min(value, 2)


def _nonnegative_count(state: AgentState, key: str) -> int:
    state_values = cast(Mapping[str, object], state)
    value = state_values.get(key)
    return value if isinstance(value, int) and value >= 0 else 0


def _response_from_state(state: AgentState) -> AgentResponse:
    final_answer = state.get("final_answer")
    if not isinstance(final_answer, str) or not final_answer:
        raise RuntimeError("Completed graph checkpoint has no final answer")
    evidence = [EvidenceItem.model_validate(item) for item in state.get("evidence", [])]
    cited_ids = cited_artifact_ids(final_answer)
    sources = [
        EvidenceReference(
            artifact_id=item.artifact_id,
            title=item.title,
            score=item.score,
            snippet=item.snippet or None,
        )
        for item in evidence
        if item.artifact_id in cited_ids
    ]
    disposition = QuestionDisposition(state["question_disposition"])
    insufficient = disposition is QuestionDisposition.KNOWLEDGE_QUESTION and (
        not bool(evidence)
        or not state.get("evidence_sufficient", False)
        or not state.get("grounding_valid", False)
    )
    input_tokens = _nonnegative_count(state, "input_tokens")
    output_tokens = _nonnegative_count(state, "output_tokens")
    return AgentResponse(
        answer=final_answer,
        disposition=disposition,
        sources=sources,
        retrieved_artifact_ids=[item.artifact_id for item in evidence],
        tool_call_count=_nonnegative_count(state, "tool_call_count"),
        model_call_count=_nonnegative_count(state, "model_call_count"),
        retrieval_round_count=_nonnegative_count(state, "retrieval_round_count"),
        input_tokens=input_tokens or None,
        output_tokens=output_tokens or None,
        insufficient_evidence=insufficient,
    )


def _create_chat_model(settings: AgentRuntimeSettings, profile: AgentProfile) -> BaseChatModel:
    model_options: dict[str, Any] = {
        "api_key": settings.openai_api_key,
        "model": profile.model_name,
        # Inngest owns durable retries. Disabling nested provider retries keeps one observable owner
        # for repeated work, while the timeout bounds each individual model request.
        "max_retries": OPENAI_MAX_RETRIES,
        "timeout": OPENAI_REQUEST_TIMEOUT_SECONDS,
    }
    if profile.temperature is not None:
        model_options["temperature"] = profile.temperature
    return ChatOpenAI(**model_options)


@asynccontextmanager
async def create_question_processor(
    settings: AgentRuntimeSettings, profile: AgentProfile
) -> AsyncIterator[StreamingQuestionProcessor]:
    if not settings.knowledge_db_path.is_file():
        raise FileNotFoundError(f"Knowledge database does not exist: {settings.knowledge_db_path}")
    model = _create_chat_model(settings, profile)
    repository = SQLiteKnowledgeRepository(settings.knowledge_db_path)
    await anyio.to_thread.run_sync(repository.validate_runtime_schema)
    retrieval_tools = KnowledgeRetrievalTools(repository)
    async with AsyncPostgresSaver.from_conn_string(settings.psycopg_database_url()) as checkpointer:
        workflow_nodes = GroundedAnswerNodes(model, retrieval_tools, profile)
        # LangGraph's compiled graph is generically typed by the framework; the protocol above
        # localizes that typing gap to this construction boundary.
        graph = cast(AgentGraph, build_graph(workflow_nodes, checkpointer=checkpointer))
        yield LangGraphQuestionProcessor(graph, settings, profile)
