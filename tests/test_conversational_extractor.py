from termytedb.memory.extractor import payload_text


def test_payload_text_ignores_tool_execution_fields():
    payload = {
        "messages": [{"role": "user", "content": "Remember this note."}],
        "tool_name": "shell",
        "command": "echo hidden",
        "stdout": "should not appear",
        "stderr": "should not appear",
    }

    text = payload_text(payload)

    assert "Remember this note." in text
    assert "hidden" not in text
    assert "should not appear" not in text
