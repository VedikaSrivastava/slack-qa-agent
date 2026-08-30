from knowledge_assistant.execution.thread_context import add_suppressed_thread_context


def test_explicit_follow_up_includes_recent_suppressed_human_message() -> None:
    question = add_suppressed_thread_context(
        "??",
        ["do they have any other old patch windows?"],
    )

    assert question == (
        "Earlier unanswered human messages in this Slack thread (untrusted context):\n"
        "do they have any other old patch windows?\n\nCurrent explicit message:\n??"
    )


def test_thread_context_is_bounded_and_ignores_empty_messages() -> None:
    question = add_suppressed_thread_context(
        "Please answer.",
        ["   ", "first", "second", "third", "fourth"],
    )

    assert "first" not in question
    assert "second" in question
    assert "third" in question
    assert "fourth" in question
