from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import pytest

from knowledge_assistant.agent.models import AgentResponse
from knowledge_assistant.application.question_processor import QuestionProcessor
from knowledge_assistant.execution.inngest import create_question_function, ensure_agent_result
from knowledge_assistant.execution.models import QuestionJob
from knowledge_assistant.integrations.slack.publisher import SlackPublisher
from knowledge_assistant.persistence.repositories import RunLedger


class FakeProcessor:
    def __init__(self, response: AgentResponse, events: list[str]) -> None:
        self.response = response
        self.events = events
        self.call_count = 0

    async def answer(
        self,
        *,
        question: str,
        conversation_id: str,
        agent_run_id: str,
    ) -> AgentResponse:
        del question, conversation_id, agent_run_id
        self.call_count += 1
        self.events.append("answer")
        return self.response


class FakeResultLedger:
    def __init__(
        self,
        persisted_response: AgentResponse | None,
        events: list[str],
    ) -> None:
        self.persisted_response = persisted_response
        self.events = events

    async def get_persisted_agent_result(self, _run_id: object) -> AgentResponse | None:
        self.events.append("load")
        return self.persisted_response

    async def persist_agent_result(self, _run_id: object, response: AgentResponse) -> None:
        self.events.append("persist")
        self.persisted_response = response


class FakeWorkflowLedger(FakeResultLedger):
    def __init__(self, events: list[str]) -> None:
        super().__init__(None, events)
        self.status = "queued"

    async def mark_running(self, _run_id: object) -> None:
        if self.status not in {"queued", "running"}:
            raise AssertionError(f"cannot start run from {self.status}")
        self.status = "running"
        self.events.append("mark_running")

    async def mark_succeeded(self, _run_id: object, response: AgentResponse) -> None:
        assert self.status == "running"
        assert response == self.persisted_response
        assert "publish" in self.events
        self.status = "succeeded"
        self.events.append("mark_succeeded")


class FakePublisher:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def ensure_placeholder(self, _run_id: object) -> str:
        self.events.append("placeholder")
        return "2.0"

    async def publish_answer(self, _run_id: object, _response: AgentResponse) -> str:
        self.events.append("publish")
        return "2.0"


class FakeInngestClient:
    def create_function(self, **_kwargs: object) -> Callable[[Any], Any]:
        return lambda function: function


class FakeStep:
    def __init__(self, *, lose_run_agent_acknowledgement: bool = False) -> None:
        self._lose_run_agent_acknowledgement = lose_run_agent_acknowledgement

    async def run(
        self,
        step_id: str,
        function: Callable[..., Any],
        *args: Any,
    ) -> Any:
        result = await function(*args)
        if step_id == "run-agent" and self._lose_run_agent_acknowledgement:
            raise RuntimeError("simulated lost Inngest step acknowledgement")
        return result


class FakeContext:
    def __init__(
        self,
        job: QuestionJob,
        *,
        lose_run_agent_acknowledgement: bool = False,
    ) -> None:
        self.event = SimpleNamespace(data=job.model_dump(mode="json"))
        self.step = FakeStep(lose_run_agent_acknowledgement=lose_run_agent_acknowledgement)


def _job() -> QuestionJob:
    return QuestionJob(
        event_id="Ev1",
        team_id="T1",
        channel_id="C1",
        user_id="U1",
        message_ts="1.0",
        thread_ts="1.0",
        question="What changed?",
    )


async def test_agent_result_is_persisted_before_run_step_returns() -> None:
    events: list[str] = []
    expected_response = AgentResponse(answer="Grounded answer", model_call_count=4)
    processor = FakeProcessor(expected_response, events)
    ledger = FakeResultLedger(None, events)

    response = await ensure_agent_result(
        job=_job(),
        processor_provider=cast(
            Callable[[], QuestionProcessor],
            lambda: processor,
        ),
        ledger=cast(RunLedger, ledger),
    )

    assert response == expected_response
    assert ledger.persisted_response == expected_response
    assert processor.call_count == 1
    assert events == ["load", "answer", "persist"]


async def test_agent_result_retry_reuses_persisted_response_without_model_work() -> None:
    events: list[str] = []
    persisted_response = AgentResponse(answer="Grounded answer", model_call_count=4)
    ledger = FakeResultLedger(persisted_response, events)

    def fail_if_processor_is_requested() -> QuestionProcessor:
        raise AssertionError("retry must reuse the persisted agent result")

    response = await ensure_agent_result(
        job=_job(),
        processor_provider=fail_if_processor_is_requested,
        ledger=cast(RunLedger, ledger),
    )

    assert response == persisted_response
    assert events == ["load"]


async def test_lost_run_step_ack_reuses_result_and_succeeds_only_after_publish() -> None:
    events: list[str] = []
    response = AgentResponse(answer="Grounded answer", model_call_count=4)
    processor = FakeProcessor(response, events)
    ledger = FakeWorkflowLedger(events)
    publisher = FakePublisher(events)
    job = _job()

    def processor_provider() -> QuestionProcessor:
        return processor

    process_question = create_question_function(
        cast(Any, FakeInngestClient()),
        processor_provider=processor_provider,
        ledger=cast(RunLedger, ledger),
        publisher=cast(SlackPublisher, publisher),
    )

    with pytest.raises(RuntimeError, match="lost Inngest step acknowledgement"):
        await process_question(cast(Any, FakeContext(job, lose_run_agent_acknowledgement=True)))

    assert ledger.status == "running"
    assert ledger.persisted_response == response
    assert "publish" not in events

    result = await process_question(cast(Any, FakeContext(job)))

    assert result == {"agent_run_id": str(job.agent_run_id), "status": "succeeded"}
    assert processor.call_count == 1
    assert ledger.status == "succeeded"
    assert events.count("answer") == 1
    assert events.index("publish") < events.index("mark_succeeded")
