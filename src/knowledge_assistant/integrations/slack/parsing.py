"""Pure helpers for normalizing Slack events."""

from __future__ import annotations

import uuid
from typing import Any

from knowledge_assistant.execution.models import (
    AgentSessionStopRequest,
    FollowUpCandidateJob,
    QuestionJob,
)


def _slack_event_uuid(team_id: str, event_id: str, purpose: str) -> uuid.UUID:
    """Derive stable internal identities so Slack delivery retries cannot fork work."""

    return uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"slack-qa-agent:{team_id}:{event_id}:{purpose}",
    )


def strip_bot_mention(text: str, bot_user_id: str) -> str:
    """Remove only the authenticated bot's Slack mention and normalize whitespace."""

    return " ".join(text.replace(f"<@{bot_user_id}>", " ").split())


def contains_user_mention(text: str, user_id: str) -> bool:
    """Return whether Slack mrkdwn explicitly mentions one exact user ID."""

    return f"<@{user_id}>" in text


def parse_app_mention(
    body: dict[str, Any],
    event: dict[str, Any],
    *,
    bot_user_id: str,
) -> QuestionJob:
    """Convert a verified Slack event into the transport-neutral job model."""

    message_ts = str(event["ts"])
    # Slack threads are single-rooted; thread_ts may equal message_ts for first mentions.
    thread_ts = str(event.get("thread_ts") or message_ts)
    event_id = str(body["event_id"])
    team_id = str(body["team_id"])
    return QuestionJob(
        agent_run_id=_slack_event_uuid(team_id, event_id, "agent-run"),
        event_id=event_id,
        team_id=team_id,
        channel_id=str(event["channel"]),
        user_id=str(event["user"]),
        message_ts=message_ts,
        thread_ts=thread_ts,
        question=strip_bot_mention(str(event.get("text", "")), bot_user_id),
    )


def parse_follow_up_candidate(body: dict[str, Any], event: dict[str, Any]) -> FollowUpCandidateJob:
    """Convert an ordinary human thread reply into a durable routing candidate."""

    message_text = event.get("text")
    if not isinstance(message_text, str):
        raise ValueError("Slack thread message text must be a string")
    # Canonicalize the raw event identity before persistence so replayed events do not
    # produce distinct rows.
    event_id = str(body["event_id"])
    team_id = str(body["team_id"])
    return FollowUpCandidateJob(
        candidate_id=_slack_event_uuid(team_id, event_id, "follow-up-candidate"),
        event_id=event_id,
        team_id=team_id,
        channel_id=event["channel"],
        user_id=event["user"],
        message_ts=event["ts"],
        # Requiring this key prevents a channel root message from creating an agent conversation.
        thread_ts=event["thread_ts"],
        message_text=message_text.strip(),
    )


def parse_agent_session_stopped(
    body: dict[str, Any], event: dict[str, Any]
) -> AgentSessionStopRequest:
    """Parse Slack's native agent-session stop event without performing cancellation I/O."""

    body_team_id = body["team_id"]
    event_team_value = event.get("team_id")
    if event_team_value is not None and body_team_id != event_team_value:
        raise ValueError("Slack stop event team does not match its envelope")
    return AgentSessionStopRequest(
        event_id=body["event_id"],
        team_id=body_team_id,
        channel_id=event["channel"],
        user_id=event["user"],
        thread_ts=event["thread_ts"],
        event_ts=event["event_ts"],
        streaming_message_ts=event["streaming_message_ts"],
    )
