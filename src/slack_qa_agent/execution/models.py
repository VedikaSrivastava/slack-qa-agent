"""Validated jobs passed from Slack ingress to durable execution."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field, computed_field


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
