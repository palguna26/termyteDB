from pathlib import Path

from termytedb.evaluation import (
    evaluate_isolation_fixture,
    evaluate_reconciliation_fixture,
    evaluate_retrieval_fixture,
    evaluate_rule_fixture,
    evaluate_temporal_fixture,
)


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
    assert result["precision_at_k"] > 0


def test_isolation_evaluation_reports_no_leaks():
    result = evaluate_isolation_fixture("tests/fixtures/isolation_cases.jsonl")
    assert result["search_leaks"] == 0
    assert result["context_leaks"] == 0
    assert result["leakage_free"] == 1


def test_reconciliation_evaluation_uses_production_path():
    fixture = Path(__file__).parent / "fixtures" / "reconciliation_cases.jsonl"
    result = evaluate_reconciliation_fixture(fixture)
    assert result["cases"] == 2
    assert result["reconciliation_accuracy"] == 1.0


def test_temporal_evaluation_measures_stale_rejection():
    fixture = Path(__file__).parent / "fixtures" / "temporal_cases.jsonl"
    result = evaluate_temporal_fixture(fixture)
    assert result["cases"] == 1
    assert result["stale_memory_rejection"] == 1.0
    assert result["temporal_state_accuracy"] == 1.0
