from pathlib import Path

from termytedb.evaluation import evaluate_continuation_fixture


def test_continuation_runner_compares_production_memory_to_baselines():
    fixture = Path(__file__).parent / "fixtures" / "continuation_cases.jsonl"
    result = evaluate_continuation_fixture(fixture)
    assert result["cases"] == 2
    assert result["baselines"]["no_memory"]["completion_rate"] == 0.0
    assert result["baselines"]["termytedb"]["completion_rate"] == 1.0
    assert result["termytedb_improvement_over_previous_summary"] == 1.0
