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


def test_tool_execution_preserves_structured_trace_and_recovery():
    payload = {
        "tool_name": "shell",
        "command": "pytest -q",
        "stdout": "12 passed",
        "stderr": "Failure: one worker crashed.",
        "exit_code": 1,
        "error": "WorkerError",
        "corrective_action": "Outcome: rerun with one worker.",
    }

    text = payload_text(payload, "tool_execution")

    assert "Tool: shell" in text
    assert "Command: pytest -q" in text
    assert "Stdout: 12 passed" in text
    assert "Stderr: Failure: one worker crashed." in text
    assert "Exit code: 1" in text
    assert "Error: WorkerError" in text
    assert "Corrective action: Outcome: rerun with one worker." in text


def test_execution_fields_are_preserved_even_for_generic_event_type():
    assert payload_text({"stderr": "connection refused", "exit_code": 2}) == (
        "Stderr: connection refused\nExit code: 2"
    )


def test_tool_event_keeps_environment_context_when_it_contains_execution_output():
    text = payload_text(
        {"text": "<environment_context>Failure: compiler exited with code 1.</environment_context>"},
        "bash_command",
    )
    assert "compiler exited with code 1" in text
