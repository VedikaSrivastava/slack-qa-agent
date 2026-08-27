from typing import Any, cast
from uuid import UUID, uuid4

from slack_sdk.web.async_client import AsyncWebClient

from knowledge_assistant.agent.models import AgentResponse, EvidenceReference
from knowledge_assistant.integrations.slack.publisher import MAX_SLACK_TEXT, SlackPublisher
from knowledge_assistant.persistence.repositories import DeliveryState, RunLedger


class FakeLedger:
    def __init__(self) -> None:
        self.placeholder: str | None = None
        self.response: str | None = None

    async def get_delivery(self, _run_id: UUID) -> DeliveryState:
        return DeliveryState("C1", "1.0", self.placeholder, self.response)

    async def set_placeholder(self, _run_id: UUID, timestamp: str) -> None:
        self.placeholder = timestamp

    async def set_response(self, _run_id: UUID, timestamp: str) -> None:
        self.response = timestamp


class FakeSlackClient:
    def __init__(self) -> None:
        self.posts: list[dict[str, Any]] = []
        self.updates: list[dict[str, Any]] = []

    async def chat_postMessage(self, **kwargs: Any) -> dict[str, str]:
        self.posts.append(kwargs)
        return {"ts": "2.0"}

    async def chat_update(self, **kwargs: Any) -> dict[str, bool]:
        self.updates.append(kwargs)
        return {"ok": True}


def _publisher(client: FakeSlackClient, ledger: FakeLedger) -> SlackPublisher:
    return SlackPublisher(
        cast(AsyncWebClient, client),
        cast(RunLedger, ledger),
    )


async def test_publisher_reuses_placeholder_on_retries() -> None:
    ledger = FakeLedger()
    client = FakeSlackClient()
    publisher = _publisher(client, ledger)
    run_id = uuid4()
    response = AgentResponse(
        answer="Grounded answer [a1].",
        sources=[EvidenceReference(artifact_id="a1", title="Runbook")],
    )

    await publisher.ensure_placeholder(run_id)
    await publisher.ensure_placeholder(run_id)
    await publisher.publish_answer(run_id, response)
    await publisher.publish_answer(run_id, response)

    assert len(client.posts) == 1
    assert len(client.updates) == 1
    assert ledger.placeholder == "2.0"
    assert ledger.response == "2.0"


async def test_publisher_does_not_overwrite_an_existing_response() -> None:
    ledger = FakeLedger()
    ledger.placeholder = "2.0"
    ledger.response = "2.0"
    client = FakeSlackClient()
    publisher = _publisher(client, ledger)
    run_id = uuid4()

    timestamp = await publisher.publish_answer(run_id, AgentResponse(answer="A retry"))
    await publisher.publish_safe_error(run_id)

    assert timestamp == "2.0"
    assert client.posts == []
    assert client.updates == []


async def test_safe_error_reuses_placeholder_and_persists_response() -> None:
    ledger = FakeLedger()
    ledger.placeholder = "2.0"
    client = FakeSlackClient()
    publisher = _publisher(client, ledger)
    run_id = uuid4()

    await publisher.publish_safe_error(run_id)
    await publisher.publish_safe_error(run_id)

    assert client.posts == []
    assert len(client.updates) == 1
    assert client.updates[0]["ts"] == "2.0"
    assert ledger.response == "2.0"


async def test_posted_messages_use_stable_uuid5_client_message_ids() -> None:
    run_id = uuid4()
    first_client = FakeSlackClient()
    second_client = FakeSlackClient()

    await _publisher(first_client, FakeLedger()).ensure_placeholder(run_id)
    await _publisher(second_client, FakeLedger()).ensure_placeholder(run_id)

    first_id = UUID(first_client.posts[0]["client_msg_id"])
    second_id = UUID(second_client.posts[0]["client_msg_id"])
    assert first_id.version == 5
    assert first_id == second_id


async def test_long_answer_is_split_and_retains_escaped_sources() -> None:
    ledger = FakeLedger()
    ledger.placeholder = "2.0"
    client = FakeSlackClient()
    publisher = _publisher(client, ledger)
    response = AgentResponse(
        answer="Use <unsafe> & verify. " + ("x" * (MAX_SLACK_TEXT * 2)),
        sources=[EvidenceReference(artifact_id="a>1", title="Runbook <ops> & support")],
    )

    await publisher.publish_answer(uuid4(), response)

    messages = [
        cast(str, client.updates[0]["text"]),
        *(cast(str, post["text"]) for post in client.posts),
    ]
    assert len(messages) == 3
    assert all(len(message) <= MAX_SLACK_TEXT for message in messages)
    assert messages[0].startswith("*Answer*\n\n")
    assert all(message.startswith("*Answer (continued)*\n\n") for message in messages[1:])
    assert "*Sources*" in messages[-1]
    assert "Runbook &lt;ops&gt; &amp; support (`a&gt;1`)" in messages[-1]
    assert all("<unsafe>" not in message for message in messages)
    assert all(post["thread_ts"] == "1.0" for post in client.posts)
    assert len({post["client_msg_id"] for post in client.posts}) == len(client.posts)


async def test_short_answer_remains_a_single_message() -> None:
    ledger = FakeLedger()
    ledger.placeholder = "2.0"
    client = FakeSlackClient()
    publisher = _publisher(client, ledger)

    await publisher.publish_answer(uuid4(), AgentResponse(answer="A concise answer."))

    assert len(client.updates) == 1
    assert client.posts == []
    assert client.updates[0]["text"] == "*Answer*\n\nA concise answer."
