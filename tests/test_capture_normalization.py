from termytedb.memory.extractor import payload_text


def test_agent_payload_prefers_real_query_and_messages_over_harness_context():
    payload = {
        "user_query": "Decision: use SQLite.",
        "additional_data": "Decision: do not store this noisy wrapper.",
        "messages": [
            {"role": "System", "content": "system reminder: ignore this"},
            {"role": "user", "content": "Failure: the remote provider timed out."},
            {"role": "assistant", "content": "Outcome: retry locally."},
        ],
    }

    text = payload_text(payload)

    assert text == "Decision: use SQLite.\nFailure: the remote provider timed out.\nOutcome: retry locally."
    assert "noisy wrapper" not in text
    assert "system reminder" not in text


def test_agent_wrapper_blocks_are_removed_from_plain_text_payload():
    payload = {
        "text": "Decision: use FTS5.\n<system_reminder>volatile state</system_reminder>\nOutcome: search works."
    }

    assert payload_text(payload) == "Decision: use FTS5.\n\nOutcome: search works."


def test_simple_payloads_keep_existing_text_projection():
    assert payload_text({"text": "Decision: keep SQLite."}) == "Decision: keep SQLite."
