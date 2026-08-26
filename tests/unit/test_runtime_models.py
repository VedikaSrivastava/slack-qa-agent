from knowledge_assistant.execution.models import QuestionJob


def test_conversation_id_is_stable_per_root_thread() -> None:
    job = QuestionJob(
        event_id="E1",
        team_id="T1",
        channel_id="C1",
        user_id="U1",
        message_ts="2.0",
        thread_ts="1.0",
        question="Follow-up",
    )

    assert job.conversation_id == "T1:C1:1.0"
