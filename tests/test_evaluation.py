from pathlib import Path

from termytedb.evaluation import evaluate_reconciliation_fixture, evaluate_retrieval_fixture, evaluate_rule_fixture


def test_rule_evaluation_is_reproducible():
    fixture = Path(__file__).parent / "fixtures" / "extraction_cases.jsonl"
    result = evaluate_rule_fixture(fixture)
    assert result["cases"] == 50
    assert result["candidate_recall"] > 0


def test_retrieval_evaluation_uses_production_path():
    fixture = Path(__file__).parent / "fixtures" / "retrieval_cases.jsonl"
    result = evaluate_retrieval_fixture(fixture)
    assert result["cases"] == 4
    assert result["recall_at_k"] == 1.0
    assert result["mrr"] == 1.0


def test_reconciliation_evaluation_uses_production_path():
    fixture = Path(__file__).parent / "fixtures" / "reconciliation_cases.jsonl"
    result = evaluate_reconciliation_fixture(fixture)
    assert result["cases"] == 2
    assert result["reconciliation_accuracy"] == 1.0
