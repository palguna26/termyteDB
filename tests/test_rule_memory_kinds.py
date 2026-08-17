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
