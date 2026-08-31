from types import SimpleNamespace

import pytest

from benchmarks.longmemeval import run_benchmark
from src.memory.provider import ProviderError


class FlakyProvider:
    name = "openrouter"
    model = "test"

    def __init__(self, failures: int, message: str = "rate limited") -> None:
        self.failures = failures
        self.message = message
        self.calls = 0

    def extract(self, request, timeout_seconds=30.0, cancellation=None):
        self.calls += 1
        if self.calls <= self.failures:
            raise ProviderError(self.message, retryable=True, error_class="http_error")
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


def test_http_429_uses_long_shared_cooldown(monkeypatch):
    class RecordingPacer:
        def __init__(self):
            self.deferred = []

        def wait(self):
            return 0.0

        def defer(self, delay):
            self.deferred.append(delay)

    pacer = RecordingPacer()
    monkeypatch.setattr(run_benchmark, "_openrouter_pacer", pacer)
    inner = FlakyProvider(failures=1, message="OpenRouter returned HTTP 429")
    provider = run_benchmark.RateLimitedExtractionProvider(inner, max_retries=2, rate_limit_cooldown=60.0)

    assert provider.extract(object()) == "ok"
    assert pacer.deferred == [60.0]


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
            "Avg Retrieved Words": None,
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


def test_normalize_legacy_manifest_equals_new_manifest():
    legacy = {"mode": "retrieval", "pack_atoms": 40, "token_budget": 1500, "dataset_sha256": "abc"}
    current = {"mode": "retrieval", "retrieval_limit": 40, "dataset_sha256": "abc"}

    assert run_benchmark._normalize_manifest_for_comparison(legacy) == run_benchmark._normalize_manifest_for_comparison(current)


def test_normalize_legacy_custom_limit_mismatch():
    legacy = {"pack_atoms": 20}
    current = {"retrieval_limit": 40}

    assert run_benchmark._normalize_manifest_for_comparison(legacy) != run_benchmark._normalize_manifest_for_comparison(current)


def test_normalize_ignores_obsolete_token_budget():
    assert run_benchmark._normalize_manifest_for_comparison({"retrieval_limit": 40, "token_budget": 1}) == run_benchmark._normalize_manifest_for_comparison({"retrieval_limit": 40, "token_budget": 9999})


def test_normalize_prefers_canonical_limit_when_both_keys_exist():
    canonical = {"retrieval_limit": 30}

    assert run_benchmark._normalize_manifest_for_comparison({"retrieval_limit": 30, "pack_atoms": 20}) == run_benchmark._normalize_manifest_for_comparison(canonical)


def test_normalize_defaults_missing_limit_to_40():
    assert run_benchmark._normalize_manifest_for_comparison({"dataset_sha256": "x"})["retrieval_limit"] == 40


def test_normalize_invalid_and_string_limits():
    assert run_benchmark._normalize_manifest_for_comparison({"pack_atoms": "bad"})["retrieval_limit"] == 40
    assert run_benchmark._normalize_manifest_for_comparison({"pack_atoms": "40"})["retrieval_limit"] == 40
    assert run_benchmark._normalize_manifest_for_comparison({"retrieval_limit": "40"})["retrieval_limit"] == 40


def test_normalize_accepts_haystack_sessions_later_than_question_metadata():
    samples = run_benchmark.normalize_samples(
        [
            {
                "question_id": "same-day-ordering",
                "question": "What was decided?",
                "question_type": "single-session-user",
                "answer": "Use SQLite.",
                "answer_session_ids": ["session-2"],
                "haystack_session_ids": ["session-2", "session-1"],
                "haystack_dates": ["2023/04/10 (Mon) 15:36", "2023/04/10 (Mon) 03:02"],
                "haystack_sessions": [
                    [{"role": "user", "content": "Use SQLite."}],
                    [{"role": "assistant", "content": "Noted."}],
                ],
                "question_date": "2023/04/10 (Mon) 10:15",
            }
        ]
    )

    assert [session[0] for session in samples[0].sessions] == ["session-1", "session-2"]
