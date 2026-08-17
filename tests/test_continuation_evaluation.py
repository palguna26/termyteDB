from pathlib import Path

from termytedb.evaluation import evaluate_continuation_fixture


def test_continuation_runner_compares_production_memory_to_baselines():
    fixture = Path(__file__).parent / "fixtures" / "continuation_cases.jsonl"
    result = evaluate_continuation_fixture(fixture)
    assert result["cases"] == 2
    assert result["baselines"]["no_memory"]["completion_rate"] == 0.0
    assert result["baselines"]["termytedb"]["completion_rate"] == 1.0
    assert result["repository_fixture_verification_rate"] == 1.0
    assert result["termytedb_improvement_over_previous_summary"] == 1.0


def test_continuation_runner_rejects_unsafe_repository_paths(tmp_path):
    fixture = tmp_path / "unsafe.jsonl"
    fixture.write_text(
        '{"snapshot_id":"x","initial_task":"a","continuation_task":"b","verification":"c","events":[],"expected":"x",'
        '"repository_snapshot":{"../secret":"x"},"resulting_repository":{},"verification_tests":[{"path":"../secret","contains":"x"}]}\n',
        encoding="utf-8",
    )
    import pytest

    with pytest.raises(ValueError, match="relative and contained"):
        evaluate_continuation_fixture(fixture)


def test_continuation_runner_executes_allowlisted_verification_command(tmp_path):
    import json

    fixture = tmp_path / "executable.jsonl"
    script = 'from pathlib import Path\nassert Path("src/value.txt").read_text() == "ok\\n"\n'
    case = {
        "snapshot_id": "x", "initial_task": "a", "continuation_task": "b", "verification": "c",
        "events": ["Decision: keep the value."], "expected": "keep the value",
        "repository_snapshot": {"verify.py": script},
        "resulting_repository": {"src/value.txt": "ok\n", "verify.py": script},
        "verification_tests": [{"path": "src/value.txt", "contains": "ok"}],
        "verification_commands": [["python", "verify.py"]],
    }
    fixture.write_text(json.dumps(case) + "\n", encoding="utf-8")
    result = evaluate_continuation_fixture(fixture)
    assert result["repository_fixture_verification_rate"] == 1.0
