from termytedb.evaluation.local_metrics import LocalCase, evaluate_cases


def test_local_metrics_are_deterministic():
    scores = evaluate_cases(
        [LocalCase("where", frozenset({"sqlite"})), LocalCase("unknown", frozenset({"missing"}), answerable=False)],
        [["Decision: use SQLite."], []], [False, True], [100, 100], [2, 0],
    )
    assert scores["evidence_recall_at_5"] == 0.5
    assert scores["contradiction_leak_rate"] == 0.0
    assert scores["safe_abstention_accuracy"] == 1.0
    assert scores["token_compression_ratio"] == 0.99
