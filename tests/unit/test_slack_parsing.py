from slack_qa_agent.integrations.slack.parsing import parse_app_mention, strip_bot_mentions


def test_strip_bot_mentions_normalizes_whitespace() -> None:
    assert strip_bot_mentions("  <@U123>   what changed? ") == "what changed?"


def test_parse_app_mention_uses_root_message_as_thread() -> None:
    job = parse_app_mention(
        {"event_id": "Ev1", "team_id": "T1"},
        {
            "channel": "C1",
            "user": "U1",
            "ts": "123.456",
            "text": "<@UBOT> What happened at BlueHarbor?",
        },
    )

    assert job.thread_ts == "123.456"
    assert job.question == "What happened at BlueHarbor?"
    assert job.conversation_id == "T1:C1:123.456"
