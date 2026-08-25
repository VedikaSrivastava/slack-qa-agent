"""LangGraph construction entry point.

The initial scaffold intentionally leaves retrieval nodes unimplemented. Keeping graph
construction here makes the agent independently invokable from Slack, Inngest, and evals.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.graph import END, START, StateGraph

from slack_qa_agent.agent.state import AgentState

AgentNode = Callable[[AgentState], Awaitable[dict[str, Any]]]


def build_graph(answer_node: AgentNode) -> Any:
    """Build the smallest executable graph around an injected answer node.

    Retrieval planning, evidence grading, and bounded refinement will be added as
    explicit nodes once the database schema and eval baseline are understood.
    """

    builder = StateGraph(AgentState)
    builder.add_node("answer", answer_node)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    return builder.compile()
