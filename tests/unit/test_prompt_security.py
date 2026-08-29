"""Guards for the two prompt contracts that other code silently depends on.

Behavioural coverage lives elsewhere: injection resistance in
``test_responder_classifier`` / ``test_thread_context`` and the
``derived-robustness-prompt-injection`` eval case; verbatim operational literals and
don't-anchor-on-the-first-match in ``test_workflow_retrieval`` and the ranking eval cases.
This file only keeps the invariants whose removal would break the grounding pipeline without
failing any other test -- it deliberately does not pin prose.
"""

from knowledge_assistant.agent.prompts import SYSTEM_GROUNDING_RULES


def test_grounding_rules_frame_evidence_as_untrusted() -> None:
    # The injection-resistance contract. If this word disappears the model may start obeying
    # instructions embedded in retrieved artifacts.
    assert "untrusted" in SYSTEM_GROUNDING_RULES.lower()


def test_grounding_rules_require_artifact_id_citations() -> None:
    # `citations.py` and the `verify_grounding` node require the model to emit `[artifact_id]`
    # markers; if the prompt stops asking for them, grounding verification silently rejects
    # every answer.
    assert "[artifact_id]" in SYSTEM_GROUNDING_RULES
