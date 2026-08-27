from __future__ import annotations

import asyncio
from typing import Any, cast
from uuid import UUID, uuid4

import pytest

from knowledge_assistant.config import SlackApplicationSettings
from knowledge_assistant.execution.dispatcher import QuestionDispatcher
from knowledge_assistant.execution.models import QuestionJob
from knowledge_assistant.integrations.slack.app import create_slack_app, process_app_mention
from knowledge_assistant.persistence.repositories import RunLedger


class FakeDispatcher:
    def __init__(self, event_ids: list[str] | None = None) -> None:
        self.jobs: list[QuestionJob] = []
        self.event_ids = ["inngest-event"] if event_ids is None else event_ids

    async def enqueue(self, job: QuestionJob) -> list[str]:
        self.jobs.append(job)
        return self.event_ids


class BlockingDispatcher:
    async def enqueue(self, job: QuestionJob) -> list[str]:
        del job
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


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


def _settings() -> SlackApplicationSettings:
    return SlackApplicationSettings(
        _env_file=None,
        app_env="test",
        openai_api_key="test-key",
        slack_bot_token="xoxb-test",
        slack_signing_secret="test-signing-secret",
        database_url="postgresql+asyncpg://user:password@postgres/test",
    )


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


@pytest.mark.parametrize("subtype", ["bot_message", "message_changed", "message_deleted"])
async def test_non_standard_message_subtypes_are_ignored(subtype: str) -> None:
    async def respond(**_kwargs: str) -> None:
        raise AssertionError("ignored events must not receive a response")

    dispatcher = FakeDispatcher()
    ledger = FakeLedger()
    event = _event("<@UBOT> What changed?")
    event["subtype"] = subtype

    await process_app_mention(
        body={"event_id": "Ev1", "team_id": "T1"},
        event=event,
        respond=respond,
        dispatcher=cast(QuestionDispatcher, dispatcher),
        ledger=cast(RunLedger, ledger),
    )

    assert dispatcher.jobs == []
    assert ledger.calls == 0


async def test_missing_inngest_event_id_fails_the_slack_request() -> None:
    async def respond(**_kwargs: str) -> None:
        raise AssertionError("valid questions must not use the validation response")

    dispatcher = FakeDispatcher(event_ids=[])
    ledger = FakeLedger()

    with pytest.raises(RuntimeError, match="Inngest returned no event ID"):
        await process_app_mention(
            body={"event_id": "Ev1", "team_id": "T1"},
            event=_event("<@UBOT> What changed?"),
            respond=respond,
            dispatcher=cast(QuestionDispatcher, dispatcher),
            ledger=cast(RunLedger, ledger),
        )

    assert len(dispatcher.jobs) == 1
    assert ledger.attachments == []


async def test_durable_handoff_is_bounded_below_slack_ack_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def respond(**_kwargs: str) -> None:
        raise AssertionError("valid questions must not use the validation response")

    monkeypatch.setattr(
        "knowledge_assistant.integrations.slack.app.SLACK_DURABLE_HANDOFF_TIMEOUT_SECONDS",
        0.001,
    )
    ledger = FakeLedger()

    with pytest.raises(TimeoutError):
        await process_app_mention(
            body={"event_id": "Ev1", "team_id": "T1"},
            event=_event("<@UBOT> What changed?"),
            respond=respond,
            dispatcher=cast(QuestionDispatcher, BlockingDispatcher()),
            ledger=cast(RunLedger, ledger),
        )

    assert ledger.calls == 1
    assert ledger.attachments == []


def test_slack_app_waits_for_durable_dispatch_before_acknowledging() -> None:
    slack_app = create_slack_app(
        _settings(),
        cast(QuestionDispatcher, FakeDispatcher()),
        cast(RunLedger, FakeLedger()),
    )

    assert slack_app.process_before_response is True
