"""Idempotent Slack status and answer delivery."""

from __future__ import annotations

import uuid

from slack_sdk.web.async_client import AsyncWebClient

from knowledge_assistant.agent.models import AgentResponse
from knowledge_assistant.persistence.repositories import RunLedger

MAX_SLACK_TEXT = 4_000
_ANSWER_HEADER = "*Answer*\n\n"
_SOURCES_HEADER = "\n\n*Sources*\n\n"
_CONTINUATION_HEADER = "*Answer (continued)*\n\n"


def _escape_slack_text(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _client_message_id(run_id: uuid.UUID, delivery_kind: str) -> str:
    # Slack can deduplicate an ambiguous network retry only when every attempt reuses this ID.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"slack-qa-agent:{run_id}:{delivery_kind}"))


def _split_text(text: str, limit: int) -> list[str]:
    """Split text at readable boundaries without discarding answer content."""
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        split_at = limit
        separator_length = 0
        for separator in ("\n\n", "\n", " "):
            candidate = remaining.rfind(separator, 0, limit + 1)
            if candidate >= limit // 2:
                split_at = candidate
                separator_length = len(separator)
                break

        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at + separator_length :].lstrip()

    if remaining:
        chunks.append(remaining)
    return chunks


def _format_answer_chunks(response: AgentResponse) -> list[str]:
    escaped_answer = _escape_slack_text(response.answer)
    source_lines = [
        f"• {_escape_slack_text(source.title)} (`{_escape_slack_text(source.artifact_id)}`)"
        for source in response.sources
    ]
    sources_section = _SOURCES_HEADER + "\n".join(source_lines) if source_lines else ""
    full_text = _ANSWER_HEADER + escaped_answer + sources_section
    content_limit = MAX_SLACK_TEXT - len(_CONTINUATION_HEADER)
    content_chunks = _split_text(full_text, content_limit)
    return [
        chunk if index == 0 else _CONTINUATION_HEADER + chunk
        for index, chunk in enumerate(content_chunks)
    ]


class SlackPublisher:
    def __init__(self, client: AsyncWebClient, ledger: RunLedger) -> None:
        self._client = client
        self._ledger = ledger

    async def ensure_placeholder(self, run_id: uuid.UUID) -> str:
        delivery = await self._ledger.get_delivery(run_id)
        if delivery.response_ts:
            return delivery.response_ts
        if delivery.placeholder_ts:
            return delivery.placeholder_ts
        result = await self._client.chat_postMessage(
            channel=delivery.channel_id,
            thread_ts=delivery.thread_ts,
            text="Searching the knowledge base…",
            client_msg_id=_client_message_id(run_id, "placeholder"),
        )
        placeholder_ts = str(result["ts"])
        await self._ledger.set_placeholder(run_id, placeholder_ts)
        return placeholder_ts

    async def publish_answer(self, run_id: uuid.UUID, response: AgentResponse) -> str:
        delivery = await self._ledger.get_delivery(run_id)
        if delivery.response_ts:
            return delivery.response_ts

        messages = _format_answer_chunks(response)
        response_ts = delivery.placeholder_ts
        if response_ts:
            await self._client.chat_update(
                channel=delivery.channel_id,
                ts=response_ts,
                text=messages[0],
            )
        else:
            result = await self._client.chat_postMessage(
                channel=delivery.channel_id,
                thread_ts=delivery.thread_ts,
                text=messages[0],
                client_msg_id=_client_message_id(run_id, "answer:1"),
            )
            response_ts = str(result["ts"])

        for part_number, text in enumerate(messages[1:], start=2):
            await self._client.chat_postMessage(
                channel=delivery.channel_id,
                thread_ts=delivery.thread_ts,
                text=text,
                client_msg_id=_client_message_id(run_id, f"answer:{part_number}"),
            )

        await self._ledger.set_response(run_id, response_ts)
        return response_ts

    async def publish_safe_error(self, run_id: uuid.UUID) -> None:
        delivery = await self._ledger.get_delivery(run_id)
        if delivery.response_ts:
            return
        text = "I couldn't complete that request right now. Please try again shortly."
        if delivery.placeholder_ts:
            await self._client.chat_update(
                channel=delivery.channel_id,
                ts=delivery.placeholder_ts,
                text=text,
            )
            await self._ledger.set_response(run_id, delivery.placeholder_ts)
            return
        result = await self._client.chat_postMessage(
            channel=delivery.channel_id,
            thread_ts=delivery.thread_ts,
            text=text,
            client_msg_id=_client_message_id(run_id, "safe-error"),
        )
        await self._ledger.set_response(run_id, str(result["ts"]))
