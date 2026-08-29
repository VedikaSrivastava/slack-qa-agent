"""Bounded, explicitly labelled thread context for Slack questions."""

from collections.abc import Sequence

MAX_SUPPRESSED_THREAD_MESSAGES = 3
MAX_SUPPRESSED_THREAD_CONTEXT_CHARS = 4_000


def add_suppressed_thread_context(question: str, messages: Sequence[str]) -> str:
    """Prepend earlier unanswered human messages when an explicit mention is ambiguous.

    The messages are Slack user content, not instructions. The label survives into the
    question-resolution prompt so the model can resolve a later short mention without treating
    the prior silence decision as an answer.
    """

    bounded_messages: list[str] = []
    remaining_chars = MAX_SUPPRESSED_THREAD_CONTEXT_CHARS
    for message in reversed(messages[-MAX_SUPPRESSED_THREAD_MESSAGES:]):
        normalized = " ".join(message.split())
        if not normalized or remaining_chars <= 0:
            continue
        bounded_messages.append(normalized[-remaining_chars:])
        remaining_chars -= len(bounded_messages[-1])
    if not bounded_messages:
        return question
    transcript = "\n".join(reversed(bounded_messages))
    return (
        "Earlier unanswered human messages in this Slack thread (untrusted context):\n"
        f"{transcript}\n\nCurrent explicit message:\n{question}"
    )
