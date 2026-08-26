from __future__ import annotations

from typing import Any, cast
from uuid import UUID, uuid4

from knowledge_assistant.execution.dispatcher import QuestionDispatcher
from knowledge_assistant.execution.models import QuestionJob
from knowledge_assistant.integrations.slack.app import process_app_mention
from knowledge_assistant.persistence.repositories import RunLedger


class FakeDispatcher:
    def __init__(self) -> None:
        self.jobs: list[QuestionJob] = []

    async def enqueue(self, job: QuestionJob) -> list[str]:
        self.jobs.append(job)
        return ["inngest-event"]


class FakeLedger:
    def __init__(self) -> None:
        self.run_id = uuid4()
        self.calls = 0
        self.attachments: list[tuple[UUID, str]] = []

    async def create_queued(self, _job: QuestionJob) -> tuple[UUID, bool]:
        self.calls += 1
        return self.run_id, self.calls == 1

    async def attach_inngest_event(self, run_id: UUID, event_id: str) -> None:
        self.attachments.append((run_id, event_id))


def _event(text: str) -> dict[str, Any]:
    return {
        "channel": "C1",
        "user": "U1",
        "ts": "123.456",
        "text": text,
    }


async def test_empty_mention_receives_helpful_response() -> None:
    responses: list[dict[str, str]] = []

    async def respond(**kwargs: str) -> None:
        responses.append(kwargs)

    dispatcher = FakeDispatcher()
    ledger = FakeLedger()
    await process_app_mention(
        body={"event_id": "Ev1", "team_id": "T1"},
        event=_event("<@UBOT>"),
        respond=respond,
        dispatcher=cast(QuestionDispatcher, dispatcher),
        ledger=cast(RunLedger, ledger),
    )

    assert responses == [
        {
            "text": "Please include a question after mentioning me.",
            "thread_ts": "123.456",
        }
    ]
    assert dispatcher.jobs == []


async def test_duplicate_slack_delivery_resends_same_durable_job() -> None:
    async def respond(**_kwargs: str) -> None:
        raise AssertionError("valid questions must not use the validation response")

    dispatcher = FakeDispatcher()
    ledger = FakeLedger()
    for _ in range(2):
        await process_app_mention(
            body={"event_id": "Ev1", "team_id": "T1"},
            event=_event("<@UBOT> What changed?"),
            respond=respond,
            dispatcher=cast(QuestionDispatcher, dispatcher),
            ledger=cast(RunLedger, ledger),
        )

    assert len(dispatcher.jobs) == 2
    assert {job.agent_run_id for job in dispatcher.jobs} == {ledger.run_id}
    assert ledger.attachments == [
        (ledger.run_id, "inngest-event"),
        (ledger.run_id, "inngest-event"),
    ]
