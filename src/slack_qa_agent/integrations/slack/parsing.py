"""Pure helpers for normalizing Slack events."""

from __future__ import annotations

import re
from typing import Any

from slack_qa_agent.execution.models import QuestionJob

BOT_MENTION_PATTERN = re.compile(r"<@[A-Z0-9]+>", flags=re.IGNORECASE)


def strip_bot_mentions(text: str) -> str:
    return " ".join(BOT_MENTION_PATTERN.sub(" ", text).split())


def parse_app_mention(body: dict[str, Any], event: dict[str, Any]) -> QuestionJob:
    """Convert a verified Slack event into the transport-neutral job model."""

    message_ts = str(event["ts"])
    thread_ts = str(event.get("thread_ts") or message_ts)
    return QuestionJob(
        event_id=str(body["event_id"]),
        team_id=str(body["team_id"]),
        channel_id=str(event["channel"]),
        user_id=str(event["user"]),
        message_ts=message_ts,
        thread_ts=thread_ts,
        question=strip_bot_mentions(str(event.get("text", ""))),
    )
