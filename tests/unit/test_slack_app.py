from __future__ import annotations

import asyncio
from typing import Any, cast
from uuid import UUID

import pytest
from slack_bolt.context.async_context import AsyncBoltContext
from slack_sdk.web.async_client import AsyncWebClient

from knowledge_assistant.config import SlackApplicationSettings
from knowledge_assistant.execution.dispatcher import QuestionDispatcher
from knowledge_assistant.execution.models import (
    AgentSessionStopRequest,
    FollowUpCandidateJob,
    QuestionJob,
)
from knowledge_assistant.integrations.slack.app import (
    StartupSlackAuthorizer,
    process_agent_session_stopped,
    process_app_mention,
    process_channel_message,
)
from knowledge_assistant.integrations.slack.routing import (
    AgentSessionStopHandoff,
    FollowUpCandidateDispatcher,
    SlackRoutingPolicy,
)


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


class FakeFollowUpDispatcher:
    def __init__(self, event_ids: list[str] | None = None) -> None:
        self.candidates: list[FollowUpCandidateJob] = []
        self.event_ids = ["follow-up-event"] if event_ids is None else event_ids

    async def enqueue_candidate(self, job: FollowUpCandidateJob) -> list[str]:
        self.candidates.append(job)
        return self.event_ids


class FakeSessionStopHandoff:
    def __init__(self) -> None:
        self.requests: list[AgentSessionStopRequest] = []

    async def enqueue_stop(self, request: AgentSessionStopRequest) -> list[str]:
        self.requests.append(request)
        return ["session-stop-event"]


ALL_RUNTIME_BOT_SCOPES = (
    "app_mentions:read",
    "assistant:write",
    "channels:history",
    "chat:write",
    "groups:history",
)


class FakeAuthTestResponse(dict[str, Any]):
    def __init__(
        self,
        *,
        scopes: tuple[str, ...],
        scope_header_name: str = "x-oauth-scopes",
    ) -> None:
        super().__init__(
            team_id="T1",
            team="Test Workspace",
            user_id="U-BOT",
            user="qa-agent",
            bot_id="B1",
            url="https://example.slack.com/",
        )
        self.headers = {scope_header_name: ",".join(scopes)}


class FakeAuthClient:
    def __init__(
        self,
        *,
        scopes: tuple[str, ...] = ALL_RUNTIME_BOT_SCOPES,
        scope_header_name: str = "x-oauth-scopes",
    ) -> None:
        self.auth_test_tokens: list[str | None] = []
        self.scopes = scopes
        self.scope_header_name = scope_header_name

    async def auth_test(self, *, token: str | None = None) -> FakeAuthTestResponse:
        self.auth_test_tokens.append(token)
        return FakeAuthTestResponse(
            scopes=self.scopes,
            scope_header_name=self.scope_header_name,
        )


def _event(text: str) -> dict[str, Any]:
    return {
        "channel": "C1",
        "user": "U1",
        "ts": "123.456",
        "text": text,
    }


def _thread_event(text: str, *, user_id: str = "U2") -> dict[str, Any]:
    return {
        "channel": "C1",
        "channel_type": "channel",
        "user": user_id,
        "ts": "123.789",
        "thread_ts": "123.456",
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


def _authorizer() -> StartupSlackAuthorizer:
    return StartupSlackAuthorizer(
        AsyncWebClient(token="xoxb-test", retry_handlers=[]),
        bot_token="xoxb-test",
    )


async def test_empty_mention_receives_helpful_response() -> None:
    responses: list[dict[str, str]] = []

    async def respond(**kwargs: str) -> None:
        responses.append(kwargs)

    dispatcher = FakeDispatcher()
    for _ in range(2):
        await process_app_mention(
            body={"event_id": "Ev1", "team_id": "T1"},
            event=_event("<@UBOT>"),
            respond=respond,
            dispatcher=cast(QuestionDispatcher, dispatcher),
            bot_user_id="UBOT",
        )

    assert len(responses) == 2
    assert responses[0]["text"] == "Please include a question after mentioning me."
    assert responses[0]["thread_ts"] == "123.456"
    UUID(responses[0]["client_msg_id"])
    assert responses[1]["client_msg_id"] == responses[0]["client_msg_id"]
    assert dispatcher.jobs == []


async def test_empty_mention_response_is_bounded_below_slack_ack_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def respond(**_kwargs: str) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(
        "knowledge_assistant.integrations.slack.app.SLACK_DURABLE_HANDOFF_TIMEOUT_SECONDS",
        0.001,
    )

    with pytest.raises(TimeoutError):
        await process_app_mention(
            body={"event_id": "Ev1", "team_id": "T1"},
            event=_event("<@UBOT>"),
            respond=respond,
            dispatcher=cast(QuestionDispatcher, FakeDispatcher()),
            bot_user_id="UBOT",
        )


async def test_startup_authorizer_caches_verified_single_workspace_identity() -> None:
    client = FakeAuthClient()
    authorizer = StartupSlackAuthorizer(
        cast(AsyncWebClient, client),
        bot_token="xoxb-test",
    )

    await authorizer.initialize()
    await authorizer.initialize()
    result = await authorizer(
        context=cast(AsyncBoltContext, object()),
        enterprise_id=None,
        team_id="T1",
        user_id="U1",
    )
    other_workspace_result = await authorizer(
        context=cast(AsyncBoltContext, object()),
        enterprise_id=None,
        team_id="T2",
        user_id="U1",
    )

    assert client.auth_test_tokens == ["xoxb-test"]
    assert result is not None
    assert result.bot_user_id == "U-BOT"
    assert result.bot_scopes == list(ALL_RUNTIME_BOT_SCOPES)
    assert other_workspace_result is None


@pytest.mark.parametrize("missing_scope", ALL_RUNTIME_BOT_SCOPES)
async def test_startup_authorizer_rejects_missing_required_scope(
    missing_scope: str,
) -> None:
    client = FakeAuthClient(
        scopes=tuple(scope for scope in ALL_RUNTIME_BOT_SCOPES if scope != missing_scope)
    )
    authorizer = StartupSlackAuthorizer(
        cast(AsyncWebClient, client),
        bot_token="xoxb-test",
    )

    with pytest.raises(RuntimeError) as exc_info:
        await authorizer.initialize()

    error_message = str(exc_info.value)
    assert missing_scope in error_message
    assert "reinstall the app" in error_message
    assert "xoxb-test" not in error_message
    assert client.auth_test_tokens == ["xoxb-test"]


async def test_startup_authorizer_reads_scope_header_case_insensitively() -> None:
    client = FakeAuthClient(scope_header_name="X-OAuth-Scopes")
    authorizer = StartupSlackAuthorizer(
        cast(AsyncWebClient, client),
        bot_token="xoxb-test",
    )

    await authorizer.initialize()

    assert client.auth_test_tokens == ["xoxb-test"]


async def test_app_mention_preserves_mentions_of_other_users() -> None:
    async def respond(**_kwargs: str) -> None:
        raise AssertionError("valid questions must not use the validation response")

    dispatcher = FakeDispatcher()

    await process_app_mention(
        body={"event_id": "Ev1", "team_id": "T1"},
        event=_event("<@UBOT> What did <@U-OTHER> approve?"),
        respond=respond,
        dispatcher=cast(QuestionDispatcher, dispatcher),
        bot_user_id="UBOT",
    )

    assert len(dispatcher.jobs) == 1
    assert dispatcher.jobs[0].question == "What did <@U-OTHER> approve?"


async def test_duplicate_slack_delivery_resends_same_durable_job() -> None:
    async def respond(**_kwargs: str) -> None:
        raise AssertionError("valid questions must not use the validation response")

    dispatcher = FakeDispatcher()
    for _ in range(2):
        await process_app_mention(
            body={"event_id": "Ev1", "team_id": "T1"},
            event=_event("<@UBOT> What changed?"),
            respond=respond,
            dispatcher=cast(QuestionDispatcher, dispatcher),
            bot_user_id="UBOT",
        )

    assert len(dispatcher.jobs) == 2
    assert dispatcher.jobs[0].event_id == dispatcher.jobs[1].event_id == "Ev1"
    assert dispatcher.jobs[0].agent_run_id == dispatcher.jobs[1].agent_run_id


@pytest.mark.parametrize("subtype", ["bot_message", "message_changed", "message_deleted"])
async def test_non_standard_message_subtypes_are_ignored(subtype: str) -> None:
    async def respond(**_kwargs: str) -> None:
        raise AssertionError("ignored events must not receive a response")

    dispatcher = FakeDispatcher()
    event = _event("<@UBOT> What changed?")
    event["subtype"] = subtype

    await process_app_mention(
        body={"event_id": "Ev1", "team_id": "T1"},
        event=event,
        respond=respond,
        dispatcher=cast(QuestionDispatcher, dispatcher),
        bot_user_id="UBOT",
    )

    assert dispatcher.jobs == []


async def test_missing_inngest_event_id_fails_the_slack_request() -> None:
    async def respond(**_kwargs: str) -> None:
        raise AssertionError("valid questions must not use the validation response")

    dispatcher = FakeDispatcher(event_ids=[])
    with pytest.raises(RuntimeError, match="Inngest returned no event ID"):
        await process_app_mention(
            body={"event_id": "Ev1", "team_id": "T1"},
            event=_event("<@UBOT> What changed?"),
            respond=respond,
            dispatcher=cast(QuestionDispatcher, dispatcher),
            bot_user_id="UBOT",
        )

    assert len(dispatcher.jobs) == 1


async def test_durable_handoff_is_bounded_below_slack_ack_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def respond(**_kwargs: str) -> None:
        raise AssertionError("valid questions must not use the validation response")

    monkeypatch.setattr(
        "knowledge_assistant.integrations.slack.app.SLACK_DURABLE_HANDOFF_TIMEOUT_SECONDS",
        0.001,
    )
    with pytest.raises(TimeoutError):
        await process_app_mention(
            body={"event_id": "Ev1", "team_id": "T1"},
            event=_event("<@UBOT> What changed?"),
            respond=respond,
            dispatcher=cast(QuestionDispatcher, BlockingDispatcher()),
            bot_user_id="UBOT",
        )


async def test_explicit_only_policy_ignores_ordinary_thread_messages() -> None:
    follow_up_dispatcher = FakeFollowUpDispatcher()

    await process_channel_message(
        body={"event_id": "Ev2", "team_id": "T1"},
        event=_thread_event("Why was that window selected?"),
        follow_up_dispatcher=cast(FollowUpCandidateDispatcher, follow_up_dispatcher),
        routing_policy=SlackRoutingPolicy.EXPLICIT_MENTIONS_ONLY,
    )

    assert follow_up_dispatcher.candidates == []


async def test_follow_up_candidate_from_any_human_is_durably_enqueued() -> None:
    follow_up_dispatcher = FakeFollowUpDispatcher()

    await process_channel_message(
        body={"event_id": "Ev2", "team_id": "T1"},
        event=_thread_event("Does that also apply to staging?", user_id="U-OTHER"),
        follow_up_dispatcher=cast(FollowUpCandidateDispatcher, follow_up_dispatcher),
        routing_policy=SlackRoutingPolicy.AGENT_OWNED_THREAD_FOLLOW_UPS,
    )

    assert len(follow_up_dispatcher.candidates) == 1
    assert follow_up_dispatcher.candidates[0].user_id == "U-OTHER"
    assert follow_up_dispatcher.candidates[0].thread_ts == "123.456"
    assert follow_up_dispatcher.candidates[0].message_text == "Does that also apply to staging?"


async def test_missing_follow_up_event_id_fails_the_slack_request() -> None:
    follow_up_dispatcher = FakeFollowUpDispatcher(event_ids=[])

    with pytest.raises(RuntimeError, match="no event ID"):
        await process_channel_message(
            body={"event_id": "Ev2", "team_id": "T1"},
            event=_thread_event("Could you explain that?"),
            follow_up_dispatcher=cast(FollowUpCandidateDispatcher, follow_up_dispatcher),
            routing_policy=SlackRoutingPolicy.AGENT_OWNED_THREAD_FOLLOW_UPS,
        )

    assert len(follow_up_dispatcher.candidates) == 1


async def test_message_copy_of_explicit_mention_is_not_enqueued_as_follow_up() -> None:
    follow_up_dispatcher = FakeFollowUpDispatcher()

    await process_channel_message(
        body={"event_id": "Ev2", "team_id": "T1"},
        event=_thread_event("<@U-BOT> Could you explain that?"),
        follow_up_dispatcher=cast(FollowUpCandidateDispatcher, follow_up_dispatcher),
        routing_policy=SlackRoutingPolicy.AGENT_OWNED_THREAD_FOLLOW_UPS,
        bot_user_id="U-BOT",
    )

    assert follow_up_dispatcher.candidates == []


async def test_bot_authored_channel_message_is_suppressed_before_routing() -> None:
    follow_up_dispatcher = FakeFollowUpDispatcher()
    event = _thread_event("Automated status update")
    event["bot_id"] = "B1"

    await process_channel_message(
        body={"event_id": "Ev2", "team_id": "T1"},
        event=event,
        follow_up_dispatcher=cast(FollowUpCandidateDispatcher, follow_up_dispatcher),
        routing_policy=SlackRoutingPolicy.AGENT_OWNED_THREAD_FOLLOW_UPS,
    )

    assert follow_up_dispatcher.candidates == []


@pytest.mark.parametrize("subtype", ["bot_message", "message_changed", "thread_broadcast"])
async def test_channel_subtypes_are_suppressed_before_routing(subtype: str) -> None:
    follow_up_dispatcher = FakeFollowUpDispatcher()
    event = _thread_event("Could you explain that?")
    event["subtype"] = subtype

    await process_channel_message(
        body={"event_id": "Ev2", "team_id": "T1"},
        event=event,
        follow_up_dispatcher=cast(FollowUpCandidateDispatcher, follow_up_dispatcher),
        routing_policy=SlackRoutingPolicy.AGENT_OWNED_THREAD_FOLLOW_UPS,
    )

    assert follow_up_dispatcher.candidates == []


async def test_agent_session_stop_event_uses_isolated_handoff() -> None:
    handoff = FakeSessionStopHandoff()

    await process_agent_session_stopped(
        body={"event_id": "EvStop", "team_id": "T1"},
        event={
            "channel": "C1",
            "event_ts": "123.999",
            "streaming_message_ts": ["123.789"],
            "thread_ts": "123.456",
            "user": "U2",
        },
        handoff=cast(AgentSessionStopHandoff, handoff),
    )

    assert len(handoff.requests) == 1
    assert handoff.requests[0].conversation_id == "T1:C1:123.456"
    assert handoff.requests[0].streaming_message_ts == ("123.789",)
