from typing import Any

import inngest

from slack_qa_agent.execution.dispatcher import InngestQuestionDispatcher
from slack_qa_agent.execution.models import QuestionJob


class FakeInngest:
    def __init__(self) -> None:
        self.event: inngest.Event | None = None

    async def send(self, event: inngest.Event) -> list[str]:
        self.event = event
        return ["event-result"]


async def test_dispatcher_serializes_conversation_key_and_deterministic_id() -> None:
    client = FakeInngest()
    dispatcher = InngestQuestionDispatcher(client)  # type: ignore[arg-type]
    job = QuestionJob(
        event_id="Ev1",
        team_id="T1",
        channel_id="C1",
        user_id="U1",
        message_ts="2.0",
        thread_ts="1.0",
        question="What changed?",
    )

    result = await dispatcher.enqueue(job)

    assert result == ["event-result"]
    assert client.event is not None
    event: Any = client.event
    assert event.id == "slack-question:Ev1"
    assert event.data["conversation_id"] == "T1:C1:1.0"
