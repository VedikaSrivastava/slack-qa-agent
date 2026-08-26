"""Idempotent Slack status and answer delivery."""

from __future__ import annotations

import uuid

from slack_sdk.web.async_client import AsyncWebClient

from knowledge_assistant.agent.models import AgentResponse
from knowledge_assistant.persistence.repositories import RunLedger

MAX_SLACK_TEXT = 35_000


class SlackPublisher:
    def __init__(self, client: AsyncWebClient, ledger: RunLedger) -> None:
        self._client = client
        self._ledger = ledger

    async def ensure_placeholder(self, run_id: uuid.UUID) -> str:
        delivery = await self._ledger.get_delivery(run_id)
        if delivery.placeholder_ts:
            return delivery.placeholder_ts
        result = await self._client.chat_postMessage(
            channel=delivery.channel_id,
            thread_ts=delivery.thread_ts,
            text="Searching the knowledge base…",
        )
        timestamp = str(result["ts"])
        await self._ledger.set_placeholder(run_id, timestamp)
        return timestamp

    async def publish_answer(self, run_id: uuid.UUID, response: AgentResponse) -> str:
        delivery = await self._ledger.get_delivery(run_id)
        source_lines = [f"• {source.title} (`{source.artifact_id}`)" for source in response.sources]
        sections = ["*Answer*", response.answer]
        if source_lines:
            sections.extend(["*Sources*", "\n".join(source_lines)])
        text = "\n\n".join(sections)
        if len(text) > MAX_SLACK_TEXT:
            text = text[: MAX_SLACK_TEXT - 32].rstrip() + "\n\n_[Answer truncated]_"

        timestamp = delivery.placeholder_ts or delivery.response_ts
        if timestamp:
            await self._client.chat_update(channel=delivery.channel_id, ts=timestamp, text=text)
        else:
            result = await self._client.chat_postMessage(
                channel=delivery.channel_id,
                thread_ts=delivery.thread_ts,
                text=text,
            )
            timestamp = str(result["ts"])
        await self._ledger.set_response(run_id, timestamp)
        return timestamp

    async def publish_safe_error(self, run_id: uuid.UUID) -> None:
        delivery = await self._ledger.get_delivery(run_id)
        text = "I couldn't complete that request right now. Please try again shortly."
        if delivery.placeholder_ts:
            await self._client.chat_update(
                channel=delivery.channel_id,
                ts=delivery.placeholder_ts,
                text=text,
            )
            return
        result = await self._client.chat_postMessage(
            channel=delivery.channel_id,
            thread_ts=delivery.thread_ts,
            text=text,
        )
        await self._ledger.set_response(run_id, str(result["ts"]))
