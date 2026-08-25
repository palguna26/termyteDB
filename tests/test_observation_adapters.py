from termytedb.memory.observation import claude_code_observation, codex_observation


def test_agent_adapters_share_event_contract():
    claude = claude_code_observation("n", "c1", {"user_message": "remember this"}, session_id="s")
    codex = codex_observation("n", "x1", {"tool_name": "pytest", "exit_code": 0}, session_id="s")
    assert claude["agent_id"] == "claude-code"
    assert codex["agent_id"] == "codex"
    assert claude["stream_id"] == codex["stream_id"] == "s"
    assert claude["type"] == codex["type"] == "agent_observation"
