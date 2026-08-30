"""Transport-independent policy for deciding whether Slack messages address the agent."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, Field

from knowledge_assistant.execution.models import AgentSessionStopRequest, FollowUpCandidateJob


class SlackRoutingPolicy(StrEnum):
    """Supported policies for messages that do not explicitly mention the agent."""

    EXPLICIT_MENTIONS_ONLY = "explicit_mentions_only"
    AGENT_OWNED_THREAD_FOLLOW_UPS = "agent_owned_thread_follow_ups"


class ResponderDecision(StrEnum):
    """Structured classifier outcomes; only RESPOND authorizes interruption."""

    RESPOND = "respond"
    STAY_SILENT = "stay_silent"
    UNCERTAIN = "uncertain"


class ResponderPromptVariant(StrEnum):
    """Code-reviewed responder prompt variants used in offline routing experiments."""

    CURRENT = "current"
    LATEST_AGENT_CONTEXT = "latest_agent_context"


class RoutingReason(StrEnum):
    POLICY_REQUIRES_MENTION = "policy_requires_mention"
    NOT_THREAD_REPLY = "not_thread_reply"
    NON_CHANNEL_MESSAGE = "non_channel_message"
    EMPTY_MESSAGE = "empty_message"
    FOLLOW_UP_CANDIDATE = "follow_up_candidate"
    CLASSIFIER_RESPOND = "classifier_respond"
    CLASSIFIER_STAY_SILENT = "classifier_stay_silent"
    CLASSIFIER_UNCERTAIN = "classifier_uncertain"
    CLASSIFIER_FAILED = "classifier_failed"


class SlackThreadIdentity(BaseModel):
    team_id: str = Field(min_length=1, max_length=128)
    channel_id: str = Field(min_length=1, max_length=128)
    thread_ts: str = Field(min_length=1, max_length=64)

    @property
    def conversation_id(self) -> str:
        return f"{self.team_id}:{self.channel_id}:{self.thread_ts}"


class ResponderClassificationRequest(BaseModel):
    """Minimal semantic input selected from durable state before model routing."""

    thread: SlackThreadIdentity
    user_id: str = Field(min_length=1, max_length=128)
    message_text: str = Field(min_length=1, max_length=8_000)
    last_agent_clarification_question: str | None = Field(
        default=None,
        min_length=1,
        max_length=8_000,
    )
    last_agent_response: str | None = Field(
        default=None,
        min_length=1,
        max_length=8_000,
    )


class ResponderClassification(BaseModel):
    """Schema suitable for structured model output."""

    decision: ResponderDecision


class SlackMessageRoutingRequest(BaseModel):
    thread: SlackThreadIdentity
    user_id: str = Field(min_length=1, max_length=128)
    message_text: str = Field(max_length=8_000)
    channel_type: str | None = Field(default=None, max_length=64)
    is_thread_reply: bool


class ResponderClassifier(Protocol):
    """Semantic boundary for classifying an already-owned thread message."""

    async def classify(
        self, request: ResponderClassificationRequest
    ) -> ResponderClassification: ...


class AgentSessionStopHandoff(Protocol):
    """Durable execution boundary for a verified Slack session-stop request."""

    async def enqueue_stop(self, request: AgentSessionStopRequest) -> list[str]: ...


class FollowUpCandidateDispatcher(Protocol):
    """Durable queue boundary used by Slack ingress before any ownership or model I/O."""

    async def enqueue_candidate(self, job: FollowUpCandidateJob) -> list[str]: ...


class RoutingAction(StrEnum):
    RESPOND = "respond"
    ENQUEUE_CANDIDATE = "enqueue_candidate"
    STAY_SILENT = "stay_silent"


@dataclass(frozen=True, slots=True)
class SlackRoutingDecision:
    action: RoutingAction
    reason: RoutingReason

    @property
    def should_respond(self) -> bool:
        return self.action is RoutingAction.RESPOND

    @property
    def should_enqueue_candidate(self) -> bool:
        return self.action is RoutingAction.ENQUEUE_CANDIDATE


def _stay_silent(reason: RoutingReason) -> SlackRoutingDecision:
    """Build an explicit no-op decision to keep call sites simple."""
    return SlackRoutingDecision(action=RoutingAction.STAY_SILENT, reason=reason)


def decide_slack_message_route(
    request: SlackMessageRoutingRequest,
    *,
    policy: SlackRoutingPolicy,
) -> SlackRoutingDecision:
    """Apply ingress-only routing without database or model calls in Slack's ack path."""

    if policy is SlackRoutingPolicy.EXPLICIT_MENTIONS_ONLY:
        # This policy is intentionally strict for noisy channels until follow-up support is
        # explicitly enabled by deployment configuration.
        return _stay_silent(RoutingReason.POLICY_REQUIRES_MENTION)
    if not request.is_thread_reply:
        # Non-thread replies are still visible to the app but are intentionally ignored by the
        # durable follow-up classifier to avoid accidental cross-topic triggers.
        return _stay_silent(RoutingReason.NOT_THREAD_REPLY)
    if request.channel_type not in {"channel", "group"}:
        # Only public channel and private group threads can be owned by the bot session model.
        return _stay_silent(RoutingReason.NON_CHANNEL_MESSAGE)
    if not request.message_text.strip():
        # Empty messages cannot produce reliable follow-up intent.
        return _stay_silent(RoutingReason.EMPTY_MESSAGE)
    return SlackRoutingDecision(
        action=RoutingAction.ENQUEUE_CANDIDATE,
        reason=RoutingReason.FOLLOW_UP_CANDIDATE,
    )


def decide_responder_classification(
    classification: ResponderClassification | None,
) -> SlackRoutingDecision:
    """Translate durable classifier output; missing or uncertain output is always silent."""

    if classification is None:
        return _stay_silent(RoutingReason.CLASSIFIER_FAILED)
    if classification.decision is ResponderDecision.RESPOND:
        return SlackRoutingDecision(
            action=RoutingAction.RESPOND,
            reason=RoutingReason.CLASSIFIER_RESPOND,
        )
    if classification.decision is ResponderDecision.UNCERTAIN:
        return _stay_silent(RoutingReason.CLASSIFIER_UNCERTAIN)
    return _stay_silent(RoutingReason.CLASSIFIER_STAY_SILENT)
