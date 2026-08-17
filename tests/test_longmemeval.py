from pathlib import Path

from termytedb.evaluation import evaluate_longmemeval_fixture


def test_longmemeval_adapter_freezes_config_and_predictions():
    fixture = Path(__file__).parent / "fixtures" / "longmemeval_cases.jsonl"
    result = evaluate_longmemeval_fixture(fixture)
    assert result["config"]["dataset_revision"] == "local-fixture"
    assert len(result["config"]["prompt_hash"]) == 64
    assert result["cases"] == 2
    assert result["accuracy"] == 1.0
    assert len(result["predictions"]) == 2
