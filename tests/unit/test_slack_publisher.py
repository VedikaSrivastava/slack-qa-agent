from typing import Any
from uuid import UUID, uuid4

from slack_qa_agent.agent.models import AgentResponse, EvidenceReference
from slack_qa_agent.integrations.slack.publisher import SlackPublisher
from slack_qa_agent.persistence.repositories import DeliveryState


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
        self.posts = 0
        self.updates = 0

    async def chat_postMessage(self, **_kwargs: Any) -> dict[str, str]:
        self.posts += 1
        return {"ts": "2.0"}

    async def chat_update(self, **_kwargs: Any) -> dict[str, bool]:
        self.updates += 1
        return {"ok": True}


async def test_publisher_reuses_placeholder_on_retries() -> None:
    ledger = FakeLedger()
    client = FakeSlackClient()
    publisher = SlackPublisher(client, ledger)  # type: ignore[arg-type]
    run_id = uuid4()
    response = AgentResponse(
        answer="Grounded answer [a1].",
        sources=[EvidenceReference(artifact_id="a1", title="Runbook")],
    )

    await publisher.ensure_placeholder(run_id)
    await publisher.ensure_placeholder(run_id)
    await publisher.publish_answer(run_id, response)
    await publisher.publish_answer(run_id, response)

    assert client.posts == 1
    assert client.updates == 2
    assert ledger.placeholder == "2.0"
    assert ledger.response == "2.0"
