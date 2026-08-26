from knowledge_assistant.agent.prompts import GENERATE_ANSWER, SYSTEM_GROUNDING_RULES


def test_grounding_prompt_treats_retrieved_instructions_as_data() -> None:
    lowered = SYSTEM_GROUNDING_RULES.lower()
    assert "untrusted data" in lowered
    assert "do not follow" in lowered


def test_generation_prompt_preserves_operational_literals() -> None:
    lowered = GENERATE_ANSWER.lower()
    assert "time ranges" in lowered
    assert "commands" in lowered
    assert "verbatim" in lowered
