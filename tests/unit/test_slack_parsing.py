import pytest

from knowledge_assistant.integrations.slack.parsing import (
    contains_user_mention,
    parse_agent_session_stopped,
    parse_app_mention,
    parse_follow_up_candidate,
    strip_bot_mention,
)


def test_strip_bot_mention_normalizes_whitespace() -> None:
    assert strip_bot_mention("  <@U123>   what changed? ", "U123") == "what changed?"


def test_strip_bot_mention_preserves_mentions_of_other_users() -> None:
    text = "<@U-BOT> ask <@U-HUMAN> what changed"

    assert strip_bot_mention(text, "U-BOT") == "ask <@U-HUMAN> what changed"


def test_contains_user_mention_matches_only_exact_bot_id() -> None:
    text = "<@U-HUMAN> could you ask <@U-BOT> about this?"

    assert contains_user_mention(text, "U-BOT") is True
    assert contains_user_mention(text, "U-OTHER") is False


def test_parse_app_mention_uses_root_message_as_thread() -> None:
    job = parse_app_mention(
        {"event_id": "Ev1", "team_id": "T1"},
        {
            "channel": "C1",
            "user": "U1",
            "ts": "123.456",
            "text": "<@UBOT> What happened at BlueHarbor?",
        },
        bot_user_id="UBOT",
    )

    assert job.thread_ts == "123.456"
    assert job.question == "What happened at BlueHarbor?"
    assert job.conversation_id == "T1:C1:123.456"


def test_parse_channel_message_requires_and_preserves_thread_root() -> None:
    candidate = parse_follow_up_candidate(
        {"event_id": "Ev2", "team_id": "T1"},
        {
            "channel": "C1",
            "user": "U2",
            "ts": "123.789",
            "thread_ts": "123.456",
            "text": "Does this also apply to staging?",
        },
    )

    assert candidate.user_id == "U2"
    assert candidate.message_ts == "123.789"
    assert candidate.thread_ts == "123.456"
    assert candidate.message_text == "Does this also apply to staging?"


def test_parse_channel_root_message_is_rejected() -> None:
    with pytest.raises(KeyError, match="thread_ts"):
        parse_follow_up_candidate(
            {"event_id": "Ev2", "team_id": "T1"},
            {
                "channel": "C1",
                "user": "U2",
                "ts": "123.789",
                "text": "Could the agent answer this?",
            },
        )


def test_parse_agent_session_stopped_validates_conversation_identity() -> None:
    request = parse_agent_session_stopped(
        {"event_id": "EvStop", "team_id": "T1"},
        {
            "channel": "C1",
            "event_ts": "123.999",
            "streaming_message_ts": ["123.789", "123.790"],
            "thread_ts": "123.456",
            "type": "agent_session_stopped",
            "user": "U2",
        },
    )

    assert request.conversation_id == "T1:C1:123.456"
    assert request.streaming_message_ts == ("123.789", "123.790")


def test_parse_agent_session_stopped_accepts_current_empty_stream_list() -> None:
    request = parse_agent_session_stopped(
        {"event_id": "EvStop", "team_id": "T1"},
        {
            "channel": "C1",
            "event_ts": "123.999",
            "streaming_message_ts": [],
            "thread_ts": "123.456",
            "type": "agent_session_stopped",
            "user": "U2",
        },
    )

    assert request.team_id == "T1"
    assert request.streaming_message_ts == ()


def test_parse_agent_session_stopped_rejects_team_mismatch() -> None:
    with pytest.raises(ValueError, match="team"):
        parse_agent_session_stopped(
            {"event_id": "EvStop", "team_id": "T1"},
            {
                "channel": "C1",
                "event_ts": "123.999",
                "streaming_message_ts": [],
                "team_id": "T2",
                "thread_ts": "123.456",
                "user": "U2",
            },
        )
