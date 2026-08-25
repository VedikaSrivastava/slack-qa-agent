"""Messages passed from Slack ingress to durable execution."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QuestionJob(BaseModel):
    event_id: str = Field(min_length=1, max_length=512)
    team_id: str = Field(min_length=1, max_length=128)
    channel_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=128)
    message_ts: str = Field(min_length=1, max_length=64)
    thread_ts: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=8_000)

    @property
    def conversation_id(self) -> str:
        return f"{self.team_id}:{self.channel_id}:{self.thread_ts}"
