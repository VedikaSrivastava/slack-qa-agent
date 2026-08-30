from __future__ import annotations

import asyncio

from knowledge_assistant.agent.models import QuestionDisposition
from knowledge_assistant.persistence import checkpoints
from knowledge_assistant.persistence.checkpoint_serialization import create_checkpoint_serializer


def test_checkpoint_loop_factory_uses_selector_loop_on_windows() -> None:
    loop = checkpoints._build_checkpoint_loop("win32")

    try:
        assert isinstance(loop, asyncio.SelectorEventLoop)
    finally:
        loop.close()


def test_checkpoint_serializer_round_trips_allowlisted_agent_state_type() -> None:
    serializer = create_checkpoint_serializer()
    encoded = serializer.dumps_typed(QuestionDisposition.KNOWLEDGE_QUESTION)

    assert serializer.loads_typed(encoded) is QuestionDisposition.KNOWLEDGE_QUESTION
