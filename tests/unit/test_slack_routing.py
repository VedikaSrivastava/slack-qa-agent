from __future__ import annotations

import pytest

from knowledge_assistant.integrations.slack.routing import (
    ResponderClassification,
    ResponderDecision,
    RoutingAction,
    RoutingReason,
    SlackMessageRoutingRequest,
    SlackRoutingPolicy,
    SlackThreadIdentity,
    decide_responder_classification,
    decide_slack_message_route,
)


def _request(
    *,
    text: str = "Why was that window selected?",
    channel_type: str | None = "channel",
    is_thread_reply: bool = True,
    user_id: str = "U2",
) -> SlackMessageRoutingRequest:
    return SlackMessageRoutingRequest(
        thread=SlackThreadIdentity(
            team_id="T1",
            channel_id="C1",
            thread_ts="123.456",
        ),
        user_id=user_id,
        message_text=text,
        channel_type=channel_type,
        is_thread_reply=is_thread_reply,
    )


def test_explicit_only_policy_stays_silent_for_ordinary_messages() -> None:
    decision = decide_slack_message_route(
        _request(),
        policy=SlackRoutingPolicy.EXPLICIT_MENTIONS_ONLY,
    )

    assert decision.action is RoutingAction.STAY_SILENT
    assert decision.reason is RoutingReason.POLICY_REQUIRES_MENTION


@pytest.mark.parametrize(
    ("routing_request", "reason"),
    [
        (_request(is_thread_reply=False), RoutingReason.NOT_THREAD_REPLY),
        (_request(channel_type="im"), RoutingReason.NON_CHANNEL_MESSAGE),
        (_request(text="   "), RoutingReason.EMPTY_MESSAGE),
    ],
)
def test_deterministic_suppressions_precede_deferred_routing(
    routing_request: SlackMessageRoutingRequest,
    reason: RoutingReason,
) -> None:
    decision = decide_slack_message_route(
        routing_request,
        policy=SlackRoutingPolicy.AGENT_OWNED_THREAD_FOLLOW_UPS,
    )

    assert decision.action is RoutingAction.STAY_SILENT
    assert decision.reason is reason


def test_valid_human_thread_reply_becomes_durable_candidate() -> None:
    decision = decide_slack_message_route(
        _request(user_id="U-SOMEONE-ELSE"),
        policy=SlackRoutingPolicy.AGENT_OWNED_THREAD_FOLLOW_UPS,
    )

    assert decision.action is RoutingAction.ENQUEUE_CANDIDATE
    assert decision.reason is RoutingReason.FOLLOW_UP_CANDIDATE


@pytest.mark.parametrize(
    ("classification", "action", "reason"),
    [
        (
            ResponderClassification(decision=ResponderDecision.RESPOND),
            RoutingAction.RESPOND,
            RoutingReason.CLASSIFIER_RESPOND,
        ),
        (
            ResponderClassification(decision=ResponderDecision.STAY_SILENT),
            RoutingAction.STAY_SILENT,
            RoutingReason.CLASSIFIER_STAY_SILENT,
        ),
        (
            ResponderClassification(decision=ResponderDecision.UNCERTAIN),
            RoutingAction.STAY_SILENT,
            RoutingReason.CLASSIFIER_UNCERTAIN,
        ),
        (None, RoutingAction.STAY_SILENT, RoutingReason.CLASSIFIER_FAILED),
    ],
)
def test_durable_responder_decision_is_conservative(
    classification: ResponderClassification | None,
    action: RoutingAction,
    reason: RoutingReason,
) -> None:
    decision = decide_responder_classification(classification)

    assert decision.action is action
    assert decision.reason is reason
