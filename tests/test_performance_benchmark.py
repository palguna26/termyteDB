from termytedb.evaluation import run_performance_benchmark


def test_performance_benchmark_reports_local_operations_and_restart():
    result = run_performance_benchmark(5)
    assert result["event_count"] == 5
    assert result["batch_events_per_second"] > 0
    assert result["recovered_jobs"] == 6
    assert result["restart_search_ms"] >= 0
    assert result["concurrent_namespace_count"] == 4
    assert result["concurrent_namespace_jobs"] == 20
    assert result["job_throughput_per_second"] > 0
    assert result["search_p95_ms"] >= 0
    assert result["context_p95_ms"] >= 0
    assert result["storage_bytes"] > 0
