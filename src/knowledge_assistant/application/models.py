"""Cross-cutting request context propagated through adapters and the agent."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RunContext(BaseModel):
    agent_run_id: str = Field(min_length=1, max_length=512)
    request_id: str = Field(min_length=1, max_length=512)
    conversation_id: str = Field(min_length=1, max_length=512)
    slack_event_id: str | None = Field(default=None, max_length=512)
    inngest_event_id: str | None = Field(default=None, max_length=512)
    prompt_version: str = Field(min_length=1, max_length=128)
    retrieval_version: str = Field(min_length=1, max_length=128)
