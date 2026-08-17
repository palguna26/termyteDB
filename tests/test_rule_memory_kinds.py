import pytest

from termytedb.extractor import extract


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Outcome: the test passed.", "outcome"),
        ("Constraint: offline mode is required.", "constraint"),
        ("Procedure: restart the worker.", "procedure"),
        ("Attempt: run the migration.", "attempt"),
        ("Task: pending review.", "task_state"),
        ("Question: which port is used?", "question"),
    ],
)
def test_rule_extractor_preserves_declared_memory_kind(text, kind):
    candidates = extract({"text": text})
    assert [candidate.kind for candidate in candidates] == [kind]
    assert text[candidates[0].start_offset : candidates[0].end_offset] == text


def test_rule_extractor_accepts_conservative_declarative_facts():
    candidates = extract({"text": "The service runs on SQLite."})
    assert len(candidates) == 1
    assert candidates[0].kind == "fact"


def test_rule_extractor_does_not_turn_unsupported_prose_into_a_fact():
    assert extract({"text": "Maybe the cache is faster."}) == []
    assert extract({"text": "The log says started."}) == []
