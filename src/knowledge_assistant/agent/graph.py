"""Bounded LangGraph construction with one retrieval refinement and one repair."""

from __future__ import annotations

from typing import Any, cast

from langgraph.graph import END, START, StateGraph

from knowledge_assistant.agent.state import AgentState
from knowledge_assistant.agent.workflow_nodes import GroundedAnswerNodes


def build_graph(nodes: GroundedAnswerNodes, *, checkpointer: Any = None) -> Any:
    builder = StateGraph(AgentState)
    builder.add_node("resolve_question", cast(Any, nodes.resolve_question))
    builder.add_node("plan_retrieval", cast(Any, nodes.plan_retrieval))
    builder.add_node("execute_retrieval", cast(Any, nodes.execute_retrieval))
    builder.add_node("grade_evidence", cast(Any, nodes.grade_evidence))
    builder.add_node("refine_retrieval", cast(Any, nodes.refine_retrieval))
    builder.add_node("generate_answer", cast(Any, nodes.generate_answer))
    builder.add_node("verify_grounding", cast(Any, nodes.verify_grounding))
    builder.add_node("repair_answer", cast(Any, nodes.repair_answer))
    builder.add_node("verify_repair", cast(Any, nodes.verify_grounding))
    builder.add_node("reject_ungrounded_answer", cast(Any, nodes.reject_ungrounded_answer))
    builder.add_node("finalize", cast(Any, nodes.finalize))

    builder.add_edge(START, "resolve_question")
    builder.add_edge("resolve_question", "plan_retrieval")
    builder.add_edge("plan_retrieval", "execute_retrieval")
    builder.add_edge("execute_retrieval", "grade_evidence")
    builder.add_conditional_edges(
        "grade_evidence",
        nodes.route_after_grade,
        {"refine": "refine_retrieval", "generate": "generate_answer"},
    )
    builder.add_edge("refine_retrieval", "execute_retrieval")
    builder.add_edge("generate_answer", "verify_grounding")
    builder.add_conditional_edges(
        "verify_grounding",
        nodes.route_after_verify,
        {"finalize": "finalize", "repair": "repair_answer"},
    )
    builder.add_edge("repair_answer", "verify_repair")
    builder.add_conditional_edges(
        "verify_repair",
        nodes.route_after_verify,
        {"finalize": "finalize", "repair": "reject_ungrounded_answer"},
    )
    builder.add_edge("reject_ungrounded_answer", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer)
