"""LangGraph-backed implementation of the application QuestionProcessor."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

import anyio
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from knowledge_assistant.agent.graph import build_graph
from knowledge_assistant.agent.models import AgentResponse, EvidenceReference
from knowledge_assistant.agent.profiles import AgentProfile
from knowledge_assistant.agent.retrieval_tools import KnowledgeRetrievalTools
from knowledge_assistant.agent.workflow_nodes import GroundedAnswerNodes
from knowledge_assistant.application.question_processor import QuestionProcessor
from knowledge_assistant.config import AgentRuntimeSettings
from knowledge_assistant.retrieval.repository import SQLiteKnowledgeRepository


class LangGraphQuestionProcessor:
    def __init__(self, graph: Any, settings: AgentRuntimeSettings, profile: AgentProfile) -> None:
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
        initial: dict[str, Any] = {
            "question": question,
            "standalone_question": question,
            "agent_run_id": agent_run_id,
            "conversation_id": conversation_id,
            "search_queries": [],
            "account_lookup": None,
            "evidence": [],
            "retrieval_round_count": 0,
            "tool_call_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "repair_attempted": False,
        }
        usage_callback = UsageMetadataCallbackHandler()
        result = cast(
            dict[str, Any],
            await self._graph.ainvoke(
                initial,
                config={
                    "callbacks": [usage_callback],
                    "configurable": {"thread_id": conversation_id},
                    "recursion_limit": 20,
                    "metadata": {
                        "agent_run_id": agent_run_id,
                        "conversation_id": conversation_id,
                        "prompt_version": self._settings.prompt_version,
                        "retrieval_version": self._settings.retrieval_version,
                        "model": self._profile.model_name,
                        "agent_profile": self._profile.name,
                        "environment": self._settings.app_env,
                        "application_version": self._settings.app_version,
                    },
                },
            ),
        )
        input_tokens = sum(
            int(usage.get("input_tokens", 0)) for usage in usage_callback.usage_metadata.values()
        )
        output_tokens = sum(
            int(usage.get("output_tokens", 0)) for usage in usage_callback.usage_metadata.values()
        )
        evidence = result.get("evidence", [])
        cited_ids = set(re.findall(r"\[([^\]\s]+)\]", str(result["final_answer"])))
        sources = [
            EvidenceReference(
                artifact_id=str(item["artifact_id"]),
                title=str(item["title"]),
                score=float(item["score"]) if item.get("score") is not None else None,
                snippet=str(item.get("snippet") or "")[:500] or None,
            )
            for item in evidence
            if str(item["artifact_id"]) in cited_ids
        ]
        insufficient = not bool(evidence) or not result.get("evidence_sufficient", False)
        return AgentResponse(
            answer=str(result["final_answer"]),
            sources=sources,
            tool_call_count=int(result.get("tool_call_count", 0)),
            retrieval_round_count=int(result.get("retrieval_round_count", 0)),
            input_tokens=input_tokens or None,
            output_tokens=output_tokens or None,
            insufficient_evidence=insufficient,
        )


@asynccontextmanager
async def create_question_processor(
    settings: AgentRuntimeSettings, profile: AgentProfile
) -> AsyncIterator[QuestionProcessor]:
    if not settings.knowledge_db_path.is_file():
        raise FileNotFoundError(f"Knowledge database does not exist: {settings.knowledge_db_path}")
    model_options: dict[str, Any] = {
        "api_key": settings.openai_api_key,
        "model": profile.model_name,
    }
    if profile.temperature is not None:
        model_options["temperature"] = profile.temperature
    model = ChatOpenAI(**model_options)
    repository = SQLiteKnowledgeRepository(settings.knowledge_db_path)
    await anyio.to_thread.run_sync(repository.validate_runtime_schema)
    retrieval_tools = KnowledgeRetrievalTools(repository)
    async with AsyncPostgresSaver.from_conn_string(settings.psycopg_database_url()) as checkpointer:
        workflow_nodes = GroundedAnswerNodes(model, retrieval_tools, profile)
        graph = build_graph(workflow_nodes, checkpointer=checkpointer)
        yield LangGraphQuestionProcessor(graph, settings, profile)
