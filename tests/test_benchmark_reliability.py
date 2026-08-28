from types import SimpleNamespace

import pytest

from benchmarks.longmemeval import run_benchmark
from src.memory.provider import ProviderError


class FlakyProvider:
    name = "openrouter"
    model = "test"

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def extract(self, request, timeout_seconds=30.0, cancellation=None):
        self.calls += 1
        if self.calls <= self.failures:
            raise ProviderError("rate limited", retryable=True, error_class="http_error")
        return "ok"


def test_request_pacer_spaces_calls():
    pacer = run_benchmark.RequestPacer(0.02)

    assert pacer.wait() == 0.0
    assert pacer.wait() >= 0.01


def test_extraction_provider_retries_retryable_failures(monkeypatch):
    monkeypatch.setattr(run_benchmark, "_openrouter_pacer", run_benchmark.RequestPacer(0.0))
    monkeypatch.setattr(run_benchmark.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(run_benchmark.random, "uniform", lambda _left, _right: 0.0)
    inner = FlakyProvider(failures=2)
    provider = run_benchmark.RateLimitedExtractionProvider(inner, max_retries=3)

    assert provider.extract(object()) == "ok"
    assert inner.calls == 3


def test_extraction_provider_does_not_retry_permanent_failure(monkeypatch):
    monkeypatch.setattr(run_benchmark, "_openrouter_pacer", run_benchmark.RequestPacer(0.0))
    inner = FlakyProvider(failures=1)
    provider = run_benchmark.RateLimitedExtractionProvider(inner, max_retries=5)
    inner.extract = lambda *args, **kwargs: (_ for _ in ()).throw(
        ProviderError("invalid output", retryable=False, error_class="invalid_output")
    )
    with pytest.raises(ProviderError, match="invalid output"):
        provider.extract(object())
    assert inner.calls == 0


def test_failed_only_scores_are_unavailable():
    sample = SimpleNamespace(question_id="q1", question_type="temporal-reasoning")
    trace = run_benchmark.failed_trace(sample, RuntimeError("boom"))

    rows = run_benchmark.summarize([trace], recall_k=15, judged=False)

    assert trace["status"] == "failed"
    assert rows == [
        {
            "Category": "Overall",
            "Completed": 0,
            "Recall@5 (%)": None,
            "Recall@10 (%)": None,
            "Recall@15 (%)": None,
            "MRR@15": None,
            "NDCG@15": None,
            "Avg Context Tokens": None,
            "Avg Latency (ms)": None,
        }
    ]
    assert "N/A" in run_benchmark.render_table(rows)


def test_end_to_end_runs_get_fresh_work_directories(tmp_path):
    args = SimpleNamespace(work_dir=str(tmp_path))

    first = run_benchmark.resolve_work_dir(args, None, is_e2e=True)
    second = run_benchmark.resolve_work_dir(args, None, is_e2e=True)

    assert first.parent == tmp_path
    assert second.parent == tmp_path
    assert first != second


def test_resume_reuses_recorded_work_directory(tmp_path):
    prior = tmp_path / "run-existing"
    args = SimpleNamespace(work_dir=str(tmp_path))

    assert run_benchmark.resolve_work_dir(args, {"config": {"work_dir": str(prior)}}, is_e2e=True) == prior
