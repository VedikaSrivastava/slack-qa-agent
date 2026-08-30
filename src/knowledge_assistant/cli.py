"""Container-friendly CLI using the same QuestionProcessor as Slack."""

from __future__ import annotations

import argparse
import asyncio
import uuid
from collections.abc import Sequence

from knowledge_assistant.agent.processor import create_question_processor
from knowledge_assistant.agent.profiles import PRODUCTION_PROFILE
from knowledge_assistant.config import get_agent_runtime_settings


async def ask(question: str, conversation_id: str | None = None) -> int:
    settings = get_agent_runtime_settings()
    run_id = str(uuid.uuid4())
    resolved_conversation_id = conversation_id or f"cli:{run_id}"
    async with create_question_processor(settings, PRODUCTION_PROFILE) as processor:
        response = await processor.answer(
            question=question,
            conversation_id=resolved_conversation_id,
            agent_run_id=run_id,
        )
    print(response.answer)
    if response.show_sources and response.sources:
        print("\nSources")
        for source in response.sources:
            print(f"- {source.title} ({source.artifact_id})")
    print(
        f"\nActions: {response.model_call_count} model calls, "
        f"{response.tool_call_count} tool calls, "
        f"{response.retrieval_round_count} retrieval rounds"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m knowledge_assistant.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ask_parser = subparsers.add_parser("ask", help="Ask one grounded question")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--conversation-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "ask":
        return asyncio.run(ask(args.question, args.conversation_id))
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
