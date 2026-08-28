"""Validated jobs passed from Slack ingress to durable execution."""

from __future__ import annotations

import uuid
from typing import Annotated

from pydantic import BaseModel, Field, computed_field

SlackTimestamp = Annotated[str, Field(min_length=1, max_length=64)]


class QuestionJob(BaseModel):
    agent_run_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    event_id: str = Field(min_length=1, max_length=512)
    team_id: str = Field(min_length=1, max_length=128)
    channel_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=128)
    message_ts: str = Field(min_length=1, max_length=64)
    thread_ts: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=8_000)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def conversation_id(self) -> str:
        return f"{self.team_id}:{self.channel_id}:{self.thread_ts}"


class FollowUpCandidateJob(BaseModel):
    """Ordinary Slack thread reply awaiting durable ownership and responder checks."""

    candidate_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    event_id: str = Field(min_length=1, max_length=512)
    team_id: str = Field(min_length=1, max_length=128)
    channel_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=128)
    message_ts: SlackTimestamp
    thread_ts: SlackTimestamp
    message_text: str = Field(min_length=1, max_length=8_000)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def conversation_id(self) -> str:
        return f"{self.team_id}:{self.channel_id}:{self.thread_ts}"


class AgentSessionStopRequest(BaseModel):
    """Validated request to cancel work for one Slack agent session."""

    event_id: str = Field(min_length=1, max_length=512)
    team_id: str = Field(min_length=1, max_length=128)
    channel_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=128)
    thread_ts: SlackTimestamp
    event_ts: SlackTimestamp
    streaming_message_ts: tuple[SlackTimestamp, ...] = Field(max_length=100)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def conversation_id(self) -> str:
        return f"{self.team_id}:{self.channel_id}:{self.thread_ts}"


class QuestionCancellationJob(AgentSessionStopRequest):
    """Durable stop event, optionally linked to the active run it should cancel."""

    agent_run_id: uuid.UUID | None = None
    cancellation_accepted: bool = False
