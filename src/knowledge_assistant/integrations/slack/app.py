"""Slack Bolt application factory."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, cast

import structlog
from pydantic import ValidationError
from slack_bolt.async_app import AsyncApp
from slack_bolt.authorization import AuthorizeResult
from slack_bolt.authorization.async_authorize import AsyncAuthorize
from slack_bolt.context.async_context import AsyncBoltContext
from slack_sdk.web.async_client import AsyncWebClient

from knowledge_assistant.config import SlackApplicationSettings
from knowledge_assistant.execution.dispatcher import QuestionDispatcher
from knowledge_assistant.execution.models import QuestionJob
from knowledge_assistant.integrations.slack.parsing import (
    contains_user_mention,
    parse_agent_session_stopped,
    parse_app_mention,
    parse_follow_up_candidate,
    strip_bot_mention,
)
from knowledge_assistant.integrations.slack.routing import (
    AgentSessionStopHandoff,
    FollowUpCandidateDispatcher,
    SlackMessageRoutingRequest,
    SlackRoutingPolicy,
    SlackThreadIdentity,
    decide_slack_message_route,
)

logger = structlog.get_logger(__name__)
SlackResponder = Callable[..., Awaitable[Any]]
SLACK_DURABLE_HANDOFF_TIMEOUT_SECONDS = 2.0
REQUIRED_SLACK_BOT_SCOPES = (
    "app_mentions:read",
    "assistant:write",
    "channels:history",
    "chat:write",
    "groups:history",
)


class StartupSlackAuthorizer(AsyncAuthorize):
    """Validate one bot token at startup and authorize ingress from the cached result."""

    def __init__(self, client: AsyncWebClient, *, bot_token: str) -> None:
        self.client = client
        self._bot_token = bot_token
        self._authorize_result: AuthorizeResult | None = None

    async def initialize(self) -> None:
        """Fail startup for an invalid token instead of calling auth.test in Slack's ack path."""

        if self._authorize_result is not None:
            return
        auth_test_result = await self.client.auth_test(token=self._bot_token)
        scope_header = next(
            (
                str(value)
                for name, value in auth_test_result.headers.items()
                if name.casefold() == "x-oauth-scopes"
            ),
            "",
        )
        installed_scopes = {scope.strip() for scope in scope_header.split(",") if scope.strip()}
        missing_scopes = [
            scope for scope in REQUIRED_SLACK_BOT_SCOPES if scope not in installed_scopes
        ]
        if missing_scopes:
            raise RuntimeError(
                "Slack bot token is missing required OAuth scopes: "
                f"{', '.join(missing_scopes)}. Update the app manifest and reinstall the app."
            )

        authorize_result = AuthorizeResult.from_auth_test_response(
            auth_test_response=auth_test_result,
            bot_token=self._bot_token,
            bot_scopes=scope_header,
        )
        if authorize_result.team_id is None or authorize_result.bot_user_id is None:
            raise RuntimeError("Slack token must authorize one workspace bot user")
        self._authorize_result = authorize_result
        logger.info(
            "slack_authorization_initialized",
            slack_team_id=authorize_result.team_id,
            slack_bot_user_id=authorize_result.bot_user_id,
        )

    async def __call__(
        self,
        *,
        context: AsyncBoltContext,
        enterprise_id: str | None,
        team_id: str | None,
        user_id: str | None,
        actor_enterprise_id: str | None = None,
        actor_team_id: str | None = None,
        actor_user_id: str | None = None,
    ) -> AuthorizeResult | None:
        del context, enterprise_id, user_id
        del actor_enterprise_id, actor_team_id, actor_user_id
        authorize_result = self._authorize_result
        if authorize_result is None:
            raise RuntimeError("Slack authorization was not initialized before serving requests")
        if team_id != authorize_result.team_id:
            return None
        return authorize_result


def _validation_response_client_message_id(event_id: str) -> str:
    """Give Slack retries one stable identity for the empty-mention response."""

    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"slack-qa-agent:{event_id}:empty-mention"))


def _is_ignored_message(event: dict[str, Any]) -> bool:
    """Suppress bot and subtype events before parsing or semantic routing."""

    return bool(event.get("bot_id")) or event.get("subtype") is not None


async def _enqueue_question(
    job: QuestionJob,
    *,
    dispatcher: QuestionDispatcher,
) -> None:
    """Complete the bounded durable handoff before Slack receives an acknowledgement."""

    # The run row is created inside the durable turn function. Keeping PostgreSQL out of this
    # boundary avoids a committed QUEUED row when the subsequent Inngest send never succeeds.
    async with asyncio.timeout(SLACK_DURABLE_HANDOFF_TIMEOUT_SECONDS):
        event_ids = await dispatcher.enqueue(job)
        if not event_ids:
            raise RuntimeError("Inngest returned no event ID for the Slack question")
    logger.info(
        "slack_question_enqueued",
        agent_run_id=str(job.agent_run_id),
        slack_event_id=job.event_id,
        conversation_id=job.conversation_id,
        inngest_event_ids=event_ids,
    )


async def process_app_mention(
    *,
    body: dict[str, Any],
    event: dict[str, Any],
    respond: SlackResponder,
    dispatcher: QuestionDispatcher,
    bot_user_id: str,
) -> None:
    """Validate one mention and dispatch it to durable processing."""

    if _is_ignored_message(event):
        return
    if not strip_bot_mention(str(event.get("text", "")), bot_user_id):
        # Empty user payload after removing the mention is treated as a benign validation edge;
        # Slack should still get a small clarification instead of entering durable execution.
        event_id = body.get("event_id")
        if not isinstance(event_id, str) or not event_id.strip():
            logger.warning("slack_event_rejected", error_code="missing_slack_event_id")
            return
        async with asyncio.timeout(SLACK_DURABLE_HANDOFF_TIMEOUT_SECONDS):
            await respond(
                text="Please include a question after mentioning me.",
                thread_ts=str(event.get("thread_ts") or event.get("ts", "")),
                client_msg_id=_validation_response_client_message_id(event_id),
            )
        return

    try:
        job = parse_app_mention(body, event, bot_user_id=bot_user_id)
    except (KeyError, ValueError, ValidationError) as exc:
        logger.warning(
            "slack_event_rejected",
            error_code="malformed_app_mention",
            exception_class=type(exc).__name__,
        )
        return
    await _enqueue_question(
        job,
        dispatcher=dispatcher,
    )


async def process_channel_message(
    *,
    body: dict[str, Any],
    event: dict[str, Any],
    follow_up_dispatcher: FollowUpCandidateDispatcher | None,
    routing_policy: SlackRoutingPolicy,
    bot_user_id: str | None = None,
) -> None:
    """Durably enqueue an ordinary thread reply for deferred ownership and semantic routing."""

    if _is_ignored_message(event):
        return
    if routing_policy is SlackRoutingPolicy.EXPLICIT_MENTIONS_ONLY:
        return
    if not str(event.get("thread_ts", "")).strip():
        # Keep non-thread replies out of the follow-up path to avoid accidental cross-thread
        # context switches in public channels.
        return
    if not str(event.get("text", "")).strip():
        # Empty thread replies cannot produce a reliable "respond" decision.
        return
    if bot_user_id is not None and contains_user_mention(str(event["text"]), bot_user_id):
        # The app_mention subscription owns this same Slack message.
        return

    try:
        candidate = parse_follow_up_candidate(body, event)
    except (KeyError, ValueError, ValidationError) as exc:
        logger.warning(
            "slack_event_rejected",
            error_code="malformed_channel_thread_message",
            exception_class=type(exc).__name__,
        )
        return

    channel_type_value = event.get("channel_type")
    channel_type = str(channel_type_value) if channel_type_value is not None else None
    decision = decide_slack_message_route(
        SlackMessageRoutingRequest(
            thread=SlackThreadIdentity(
                team_id=candidate.team_id,
                channel_id=candidate.channel_id,
                thread_ts=candidate.thread_ts,
            ),
            user_id=candidate.user_id,
            message_text=candidate.message_text,
            channel_type=channel_type,
            is_thread_reply=True,
        ),
        policy=routing_policy,
    )
    if not decision.should_enqueue_candidate:
        logger.debug(
            "slack_thread_message_suppressed",
            slack_event_id=candidate.event_id,
            conversation_id=candidate.conversation_id,
            routing_reason=decision.reason.value,
        )
        return
    if follow_up_dispatcher is None:
        raise RuntimeError("Follow-up candidate routing is enabled without a durable dispatcher")

    async with asyncio.timeout(SLACK_DURABLE_HANDOFF_TIMEOUT_SECONDS):
        event_ids = await follow_up_dispatcher.enqueue_candidate(candidate)
        if not event_ids:
            raise RuntimeError("Inngest returned no event ID for the Slack follow-up candidate")
    logger.info(
        "slack_follow_up_candidate_enqueued",
        candidate_id=str(candidate.candidate_id),
        slack_event_id=candidate.event_id,
        conversation_id=candidate.conversation_id,
        inngest_event_ids=event_ids,
    )


async def process_agent_session_stopped(
    *,
    body: dict[str, Any],
    event: dict[str, Any],
    handoff: AgentSessionStopHandoff,
) -> None:
    """Validate and durably hand off one native Slack agent-session stop request."""

    try:
        request = parse_agent_session_stopped(body, event)
    except (KeyError, ValueError, ValidationError) as exc:
        logger.warning(
            "slack_event_rejected",
            error_code="malformed_agent_session_stopped",
            exception_class=type(exc).__name__,
        )
        return

    async with asyncio.timeout(SLACK_DURABLE_HANDOFF_TIMEOUT_SECONDS):
        event_ids = await handoff.enqueue_stop(request)
        if not event_ids:
            raise RuntimeError("Inngest returned no event ID for the Slack session stop")
    logger.info(
        "slack_agent_session_stop_enqueued",
        slack_event_id=request.event_id,
        conversation_id=request.conversation_id,
        inngest_event_ids=event_ids,
    )


def create_slack_app(
    settings: SlackApplicationSettings,
    dispatcher: QuestionDispatcher,
    *,
    authorizer: StartupSlackAuthorizer,
    routing_policy: SlackRoutingPolicy = SlackRoutingPolicy.EXPLICIT_MENTIONS_ONLY,
    follow_up_dispatcher: FollowUpCandidateDispatcher | None = None,
    session_stop_handoff: AgentSessionStopHandoff | None = None,
) -> AsyncApp:
    """Create a single-workspace Slack app with SDK-managed signature validation."""

    app = AsyncApp(
        client=authorizer.client,
        # Bolt accepts AsyncAuthorize instances at runtime, although this constructor's
        # published annotation only exposes a callable returning a non-optional result.
        authorize=cast(Callable[..., Awaitable[AuthorizeResult]], authorizer),
        signing_secret=settings.slack_signing_secret.get_secret_value(),
        raise_error_for_unhandled_request=True,
        # Slack must not receive a 2xx acknowledgment until the durable Inngest handoff succeeds.
        process_before_response=True,
    )

    @app.event("app_mention")
    async def handle_app_mention(
        body: dict[str, Any],
        event: dict[str, Any],
        say: SlackResponder,
        context: AsyncBoltContext,
    ) -> None:
        bot_user_id = context.bot_user_id
        if bot_user_id is None:
            raise RuntimeError("Slack request context has no authenticated bot user ID")
        await process_app_mention(
            body=body,
            event=event,
            respond=say,
            dispatcher=dispatcher,
            bot_user_id=bot_user_id,
        )

    @app.event("message")
    async def handle_channel_message(
        body: dict[str, Any],
        event: dict[str, Any],
        context: AsyncBoltContext,
    ) -> None:
        await process_channel_message(
            body=body,
            event=event,
            follow_up_dispatcher=follow_up_dispatcher,
            routing_policy=routing_policy,
            bot_user_id=context.bot_user_id,
        )

    if session_stop_handoff is not None:
        configured_session_stop_handoff = session_stop_handoff

        @app.event("agent_session_stopped")
        async def handle_agent_session_stopped(
            body: dict[str, Any],
            event: dict[str, Any],
        ) -> None:
            await process_agent_session_stopped(
                body=body,
                event=event,
                handoff=configured_session_stop_handoff,
            )

    return app
