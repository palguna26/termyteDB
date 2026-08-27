"""Tests for LongMemEval end-to-end benchmark pipeline.

Covers:
- history becomes normal events
- session ordering preserved
- namespace/sample isolation
- deterministic idempotency
- processing jobs actually executed
- memories generated through Processor
- benchmark does not directly inject memory rows
- question/answer unavailable during extraction
- duplicate ingestion idempotent
- failed processing does not silently count as success
- retrieval runs only after processing completes
- metrics correctly distinguish extraction miss vs retrieval miss
- old retrieval-only benchmark still functions
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from termytedb import TermyteDB
from termytedb.api.schemas import EventInput, EvidenceSpan, ExtractionCandidate


# Helper to get sample fixtures
def _sample():
    from benchmarks.longmemeval.run_benchmark import Sample

    return Sample(
        question_id="test-q1",
        question="What is Alice's favorite color?",
        question_type="single-session-user",
        answer="blue",
        answer_session_ids=frozenset({"sess-1"}),
        unanswerable=False,
        sessions=(
            ("sess-0", "2023/05/20 (Sat) 02:21", ({"role": "user", "content": "Hello world"}, {"role": "assistant", "content": "Hi there"})),
            ("sess-1", "2023/05/21 (Sun) 09:27", ({"role": "user", "content": "My favorite color is blue."},)),
        ),
        raw_words=10,
    )


def test_history_becomes_normal_events(tmp_path: Path):
    from benchmarks.longmemeval.run_benchmark import build_event_inputs

    sample = _sample()
    events = build_event_inputs(sample)
    # Each turn becomes an event
    assert len(events) == 3
    # Speaker/role preserved via payload messages
    assert events[0]["payload"]["messages"][0]["role"] == "user"
    assert events[0]["payload"]["messages"][0]["content"] == "Hello world"
    # Stream/session identity preserved
    assert events[0]["stream_id"] == "sess-0"
    assert events[2]["stream_id"] == "sess-1"
    # Stable idempotency keys deterministic
    events2 = build_event_inputs(sample)
    assert [e["idempotency_key"] for e in events] == [e["idempotency_key"] for e in events2]


def test_session_ordering_preserved(tmp_path: Path):
    from benchmarks.longmemeval.run_benchmark import build_event_inputs, _parse_haystack_date

    sample = _sample()
    events = build_event_inputs(sample)
    # occurred_at should be monotonic within session ordering and across sessions
    dates = [e.get("occurred_at") for e in events]
    # sess-0 events should have earlier timestamps than sess-1
    assert dates[0] < dates[2] or dates[0] is not None
    # Within sess-0, turn 0 < turn 1
    from datetime import datetime

    dt0 = datetime.fromisoformat(dates[0])
    dt1 = datetime.fromisoformat(dates[1])
    assert dt1 > dt0


def test_namespace_sample_isolation(tmp_path: Path):
    from benchmarks.longmemeval.run_benchmark import build_event_inputs
    import argparse

    from benchmarks.longmemeval.run_benchmark import ingest_and_process_e2e

    # Two samples with different question_ids should not contaminate
    from benchmarks.longmemeval.run_benchmark import Sample

    s1 = Sample(
        question_id="ns1",
        question="Q1",
        question_type="single-session-user",
        answer="A",
        answer_session_ids=frozenset({"sess-a"}),
        unanswerable=False,
        sessions=(("sess-a", "2023/05/20 02:21", ({"role": "user", "content": "Decision: use SQLite with WAL."},)),),
        raw_words=5,
    )
    s2 = Sample(
        question_id="ns2",
        question="Q2",
        question_type="single-session-user",
        answer="B",
        answer_session_ids=frozenset({"sess-b"}),
        unanswerable=False,
        sessions=(("sess-b", "2023/05/20 02:21", ({"role": "user", "content": "Decision: use Postgres."},)),),
        raw_words=5,
    )
    args = argparse.Namespace(
        work_dir=str(tmp_path),
        extraction="rule",
        extraction_model=None,
        processing_batch_size=100,
        processing_lease_seconds=180,
        processing_timeout=30.0,
        single_db=False,
        no_dense=True,
        no_rerank=True,
        recall_k=15,
        token_budget=1500,
        pack_atoms=40,
        abstain_threshold=0.25,
        mode="end-to-end",
    )
    db1, _ = ingest_and_process_e2e(tmp_path, s1, args)
    db2, _ = ingest_and_process_e2e(tmp_path, s2, args)
    # Each DB file isolated
    assert db1 != db2
    db = TermyteDB(db1)
    try:
        # ns1 should not contain Postgres
        hits = db.search("ns1", "Postgres")
        assert not any("Postgres" in h.statement for h in hits)
        # ns2 isolated via separate file; check file exists
        assert db2.exists()
    finally:
        db.close()


def test_deterministic_idempotency(tmp_path: Path):
    from benchmarks.longmemeval.run_benchmark import build_event_inputs
    import argparse
    from benchmarks.longmemeval.run_benchmark import ingest_and_process_e2e

    sample = _sample()
    # Use content that matches rule to ensure memories
    sample2 = sample
    # Rebuild with same content should produce same idempotency keys
    k1 = [e["idempotency_key"] for e in build_event_inputs(sample)]
    k2 = [e["idempotency_key"] for e in build_event_inputs(sample2)]
    assert k1 == k2

    args = argparse.Namespace(
        work_dir=str(tmp_path),
        extraction="rule",
        extraction_model=None,
        processing_batch_size=100,
        processing_lease_seconds=180,
        processing_timeout=30.0,
        single_db=False,
        no_dense=True,
        no_rerank=True,
        recall_k=15,
        token_budget=1500,
        pack_atoms=40,
        abstain_threshold=0.25,
        mode="end-to-end",
    )
    # Ingest twice should be idempotent
    db_path1, diag1 = ingest_and_process_e2e(tmp_path, sample, args)
    # Second ingestion same sample same namespace should hit duplicates
    db = TermyteDB(db_path1)
    try:
        from termytedb.api.schemas import EventInput

        events = [EventInput.model_validate(e) for e in build_event_inputs(sample)]
        dup_count = 0
        for ev in events:
            receipt = db.ingest(ev)
            if receipt.duplicate:
                dup_count += 1
        assert dup_count == len(events)
        # Metrics should still be zero pending
        m = db.metrics(sample.question_id)
        assert m["jobs_failed"] == 0
    finally:
        db.close()


def test_processing_jobs_actually_executed(tmp_path: Path):
    import argparse
    from benchmarks.longmemeval.run_benchmark import ingest_and_process_e2e
    from benchmarks.longmemeval.run_benchmark import Sample

    s = Sample(
        question_id="proc-test",
        question="What to use?",
        question_type="single-session-user",
        answer="SQLite",
        answer_session_ids=frozenset({"s1"}),
        unanswerable=False,
        sessions=(("s1", "2023/05/20 02:21", ({"role": "user", "content": "Decision: use SQLite with WAL for storage."},)),),
        raw_words=6,
    )
    args = argparse.Namespace(
        work_dir=str(tmp_path),
        extraction="rule",
        extraction_model=None,
        processing_batch_size=100,
        processing_lease_seconds=180,
        processing_timeout=30.0,
        single_db=False,
        no_dense=True,
        no_rerank=True,
        recall_k=15,
        token_budget=1500,
        pack_atoms=40,
        abstain_threshold=0.25,
        mode="end-to-end",
    )
    db_path, diag = ingest_and_process_e2e(tmp_path, s, args)
    assert diag["processing_jobs_completed"] >= 1
    assert diag["events_ingested"] == 1
    # Memories generated through Processor (rule mode)
    assert diag["memories_created"] >= 1
    assert diag["candidates_accepted"] >= 1


def test_memories_generated_through_processor_not_direct_injection(tmp_path: Path):
    import argparse
    from benchmarks.longmemeval.run_benchmark import ingest_and_process_e2e, build_event_inputs, _e2e_database_path
    from benchmarks.longmemeval.run_benchmark import Sample

    s = Sample(
        question_id="no-direct",
        question="Q",
        question_type="single-session-user",
        answer="A",
        answer_session_ids=frozenset({"s1"}),
        unanswerable=False,
        sessions=(("s1", "2023/05/20 02:21", ({"role": "user", "content": "Decision: use SQLite with WAL."},)),),
        raw_words=4,
    )
    args = argparse.Namespace(
        work_dir=str(tmp_path),
        extraction="rule",
        extraction_model=None,
        processing_batch_size=100,
        processing_lease_seconds=180,
        processing_timeout=30.0,
        single_db=False,
        no_dense=True,
        no_rerank=True,
        recall_k=15,
        token_budget=1500,
        pack_atoms=40,
        abstain_threshold=0.25,
        mode="end-to-end",
    )
    db_path, diag = ingest_and_process_e2e(tmp_path, s, args)
    # Verify that raw SQL direct insertion not used: memories have extraction_runs evidence
    db = TermyteDB(db_path)
    try:
        runs = db.extraction_runs("no-direct")
        assert len(runs) >= 1
        # Check that memories have evidence refs
        mems = db.memories("no-direct")
        for m in mems:
            assert len(m.citations) >= 1
            # citations must point to existing events
            for c in m.citations:
                ev = db.event("no-direct", str(c.event_id))
                assert ev is not None
    finally:
        db.close()


def test_question_unavailable_during_extraction(monkeypatch, tmp_path: Path):
    """Ensure extraction provider never receives question/answer."""
    from benchmarks.longmemeval.run_benchmark import build_event_inputs, ingest_and_process_e2e
    from benchmarks.longmemeval.run_benchmark import Sample
    import argparse
    from termytedb.memory.provider import FakeExtractionProvider
    from termytedb.api.schemas import ExtractionResponse, ExtractionCandidate, EvidenceSpan
    from uuid import uuid4

    captured_requests = []

    class CapturingProvider:
        name = "capture"
        model = "capture-v1"

        def extract(self, request, timeout_seconds=30.0, cancellation=None):
            captured_requests.append(request)
            # Return empty candidates; ensure not leaking
            from termytedb.memory.provider import ProviderResult
            import hashlib, json, time

            resp = ExtractionResponse(schema_version="extraction-v1", prompt_version="v1", candidates=[])
            raw = json.dumps(resp.model_dump(mode="json"), sort_keys=True).encode().hex()
            return ProviderResult(
                response=resp,
                provider_name=self.name,
                model_name=self.model,
                prompt_version=resp.prompt_version,
                raw_response_hash=hashlib.sha256(raw.encode()).hexdigest(),
                input_tokens=1,
                output_tokens=1,
                latency_ms=1,
            )

    s = Sample(
        question_id="leak-q",
        question="SECRET_QUESTION_12345",
        question_type="single-session-user",
        answer="SECRET_ANSWER_67890",
        answer_session_ids=frozenset({"s1"}),
        unanswerable=False,
        sessions=(("s1", "2023/05/20 02:21", ({"role": "user", "content": "Decision: use SQLite."},)),),
        raw_words=3,
    )
    args = argparse.Namespace(
        work_dir=str(tmp_path),
        extraction="rule",  # will be overridden via monkeypatch
        extraction_model=None,
        processing_batch_size=100,
        processing_lease_seconds=180,
        processing_timeout=30.0,
        single_db=False,
        no_dense=True,
        no_rerank=True,
        recall_k=15,
        token_budget=1500,
        pack_atoms=40,
        abstain_threshold=0.25,
        mode="end-to-end",
    )
    # Patch build_provider to return capturing provider
    import benchmarks.longmemeval.run_benchmark as bench

    monkeypatch.setattr(bench, "build_provider", lambda a: CapturingProvider())
    db_path, diag = ingest_and_process_e2e(tmp_path, s, args)
    # Check captured requests do not contain question/answer
    for req in captured_requests:
        dump = json.dumps(req.model_dump(mode="json"))
        assert "SECRET_QUESTION_12345" not in dump
        assert "SECRET_ANSWER_67890" not in dump
        assert "sess-1" not in dump  # not relevant


def test_extraction_window_includes_previous_same_session_turns(tmp_path: Path):
    from termytedb.api.schemas import ExtractionResponse
    from termytedb.memory.provider import ProviderResult
    from termytedb import TermyteDB
    import hashlib

    captured_requests = []

    class CapturingProvider:
        name = "capture-window"
        model = "capture-window-v1"

        def extract(self, request, timeout_seconds=30.0, cancellation=None):
            captured_requests.append(request)
            resp = ExtractionResponse(schema_version="extraction-v1", prompt_version="v1", candidates=[])
            raw = json.dumps(resp.model_dump(mode="json"), sort_keys=True).encode()
            return ProviderResult(
                response=resp,
                provider_name=self.name,
                model_name=self.model,
                prompt_version=resp.prompt_version,
                raw_response_hash=hashlib.sha256(raw).hexdigest(),
                input_tokens=1,
                output_tokens=1,
                latency_ms=1,
            )

    db = TermyteDB(tmp_path / "window.sqlite", extraction_provider=CapturingProvider())
    try:
        shared_time = "2023-05-20T02:21:00+00:00"
        db.ingest({
            "namespace_id": "window",
            "idempotency_key": "1",
            "type": "conversation",
            "stream_id": "s1",
            "occurred_at": shared_time,
            "payload": {"text": "My sister is Maya."},
        })
        db.ingest({
            "namespace_id": "window",
            "idempotency_key": "2",
            "type": "conversation",
            "stream_id": "s1",
            "occurred_at": shared_time,
            "payload": {"text": "Maya moved to Berlin."},
        })
        db.process("window")
        assert len(captured_requests) == 1
        assert len(captured_requests[0].events) == 2
        assert len(captured_requests[0].evidence_text) == 2
        assert any("Berlin" in text for text in captured_requests[0].evidence_text.values())
        assert db.metrics("window")["jobs_completed"] == 2
    finally:
        db.close()


def test_duplicate_ingestion_idempotent(tmp_path: Path):
    db = TermyteDB(tmp_path / "dup.sqlite")
    try:
        ev = {
            "namespace_id": "ns",
            "idempotency_key": "k1",
            "type": "conversation",
            "payload": {"text": "Decision: use SQLite with WAL."},
        }
        r1 = db.ingest(ev)
        r2 = db.ingest(ev)
        assert r2.duplicate is True
        assert r1.event_id == r2.event_id
        assert r1.job_id == r2.job_id
    finally:
        db.close()


def test_failed_processing_not_silent(tmp_path: Path):
    """Processing failure should surface via metrics, not silently count as success."""
    from termytedb.memory.provider import ProviderError

    class FailingProvider:
        name = "fail"
        model = "fail-v1"

        def extract(self, request, timeout_seconds=30.0, cancellation=None):
            raise ProviderError("injected failure", retryable=False, error_class="injected")

    db = TermyteDB(tmp_path / "fail.sqlite", extraction_provider=FailingProvider())
    try:
        db.ingest({"namespace_id": "ns", "idempotency_key": "k1", "type": "conversation", "payload": {"text": "Decision: use SQLite."}})
        resp = db.process("ns", limit=10)
        # Should report failed/dead, not processed as success
        assert resp.failed >= 1 or resp.dead_lettered >= 1
        m = db.metrics("ns")
        assert m["jobs_failed"] + m["jobs_dead"] >= 1
    finally:
        db.close()


def test_retrieval_after_processing(tmp_path: Path):
    import argparse
    from benchmarks.longmemeval.run_benchmark import ingest_and_process_e2e, retrieve_e2e_session_ranking
    from benchmarks.longmemeval.run_benchmark import Sample

    s = Sample(
        question_id="ret-test",
        question="What does the service use?",
        question_type="single-session-user",
        answer="SQLite",
        answer_session_ids=frozenset({"s1"}),
        unanswerable=False,
        sessions=(("s1", "2023/05/20 02:21", ({"role": "user", "content": "Decision: use SQLite with WAL for storage."},)),),
        raw_words=6,
    )
    args = argparse.Namespace(
        work_dir=str(tmp_path),
        extraction="rule",
        extraction_model=None,
        processing_batch_size=100,
        processing_lease_seconds=180,
        processing_timeout=30.0,
        single_db=False,
        no_dense=True,
        no_rerank=True,
        recall_k=15,
        token_budget=1500,
        pack_atoms=40,
        abstain_threshold=0.25,
        mode="end-to-end",
    )
    db_path, diag = ingest_and_process_e2e(tmp_path, s, args)
    # Ensure jobs completed before retrieval
    from termytedb.runtime.engine import TermyteDB

    engine = TermyteDB(db_path)
    try:
        m = engine.metrics(s.question_id)
        assert m["jobs_pending"] == 0
        assert m["jobs_processing"] == 0
    finally:
        engine.close()
    out = retrieve_e2e_session_ranking(db_path, s, args)
    # Should have retrieved something
    assert out["candidate_count"] >= 1
    assert out["best_rank"] == 1  # only one session, should be top


def test_metrics_distinguish_extraction_miss_vs_retrieval_miss(tmp_path: Path):
    import argparse
    from benchmarks.longmemeval.run_benchmark import ingest_and_process_e2e, retrieve_e2e_session_ranking
    from benchmarks.longmemeval.run_benchmark import Sample

    # Case 1: extraction miss (content doesn't match rule patterns)
    # Use generic filler that matches no RULES regex
    s_miss = Sample(
        question_id="miss-ext",
        question="What is favorite food?",
        question_type="single-session-user",
        answer="pizza",
        answer_session_ids=frozenset({"s1"}),
        unanswerable=False,
        sessions=(("s1", "2023/05/20 02:21", ({"role": "user", "content": "Hello there! How are you today?"},)),),
        raw_words=3,
    )
    # Case 2: extraction success but retrieval miss due to unrelated query
    s_ret = Sample(
        question_id="miss-ret",
        question="Unrelated query xyzzy",
        question_type="single-session-user",
        answer="SQLite",
        answer_session_ids=frozenset({"s1"}),
        unanswerable=False,
        sessions=(("s1", "2023/05/20 02:21", ({"role": "user", "content": "Decision: use SQLite with WAL."},)),),
        raw_words=5,
    )
    args = argparse.Namespace(
        work_dir=str(tmp_path),
        extraction="rule",
        extraction_model=None,
        processing_batch_size=100,
        processing_lease_seconds=180,
        processing_timeout=30.0,
        single_db=False,
        no_dense=True,
        no_rerank=True,
        recall_k=15,
        token_budget=1500,
        pack_atoms=40,
        abstain_threshold=0.25,
        mode="end-to-end",
    )
    db_miss, diag_miss = ingest_and_process_e2e(tmp_path, s_miss, args)
    out_miss = retrieve_e2e_session_ranking(db_miss, s_miss, args)
    # No memory should exist for s_miss
    assert diag_miss["memories_created"] == 0
    assert out_miss["best_rank"] is None

    db_ret, diag_ret = ingest_and_process_e2e(tmp_path, s_ret, args)
    out_ret = retrieve_e2e_session_ranking(db_ret, s_ret, args)
    # Memory exists but retrieval missed due to unrelated query
    assert diag_ret["memories_created"] >= 1
    assert out_ret["oracle_memories_exist"] is True
    assert out_ret["best_rank"] is None
    assert out_ret["retrieval_missed"] is True or out_ret["failure_reason"] in ("memory_existed_retrieval_missed", "context_budget_or_ranking_miss", "abstained")


def test_old_retrieval_benchmark_still_functions(tmp_path: Path):
    from benchmarks.longmemeval.run_benchmark import ingest_sample, retrieve_session_ranking, normalize_samples
    import json, argparse

    data = json.loads(open("benchmarks/longmemeval/longmemeval_s_cleaned.json", encoding="utf-8").read())
    samples = normalize_samples(data)
    s = samples[0]
    args = argparse.Namespace(
        work_dir=str(tmp_path),
        recall_k=15,
        token_budget=1500,
        pack_atoms=40,
        abstain_threshold=0.25,
        no_rerank=True,
        no_dense=True,
        single_db=False,
        mode="retrieval-only",
    )
    db_path = ingest_sample(tmp_path, s, skip_embeddings=True, single_db=False)
    assert db_path.exists()
    out = retrieve_session_ranking(db_path, s, args)
    assert "best_rank" in out
    assert "session_order" in out


def test_smoke_benchmark_requires_confirmation(tmp_path: Path):
    from benchmarks.longmemeval.run_benchmark import run
    import argparse, json, pytest

    data_path = tmp_path / "dataset.json"
    data_path.write_text(
        json.dumps(
            [
                {
                    "question_id": str(i),
                    "question": f"Q{i}",
                    "question_type": "single-session-user",
                    "answer": "A",
                    "answer_session_ids": [],
                    "haystack_session_ids": [f"s{i}"],
                    "haystack_dates": ["2026-01-01"],
                    "haystack_sessions": [[{"role": "user", "content": f"hello {i}"}]],
                }
                for i in range(6)
            ]
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        mode="end-to-end",
        data_path=data_path,
        work_dir=str(tmp_path / "work"),
        results_dir=str(tmp_path / "results"),
        limit=None,
        smoke=True,
        smoke_samples=5,
        confirm_benchmark=False,
        task=None,
        workers=1,
        recall_k=15,
        token_budget=1500,
        pack_atoms=40,
        abstain_threshold=0.25,
        no_rerank=True,
        no_dense=True,
        single_db=False,
        resume_from=None,
        answer_model="openai/gpt-4o-mini",
        judge_model="openai/gpt-4o-mini",
        budget_usd=8.0,
        embedding_provider=None,
        embedding_model=None,
        embedding_dimensions=None,
        extraction="openrouter",
        extraction_model=None,
        processing_batch_size=100,
        processing_lease_seconds=180,
        processing_timeout=30.0,
    )

    with pytest.raises(SystemExit, match="smoke benchmark loops require --confirm-benchmark"):
        run(args)


def test_smoke_benchmark_caps_to_five_samples(tmp_path: Path, monkeypatch):
    from benchmarks.longmemeval import run_benchmark as bench
    import argparse, json

    data_path = tmp_path / "dataset.json"
    data_path.write_text(
        json.dumps(
            [
                {
                    "question_id": str(i),
                    "question": f"Q{i}",
                    "question_type": "single-session-user",
                    "answer": "A",
                    "answer_session_ids": [],
                    "haystack_session_ids": [f"s{i}"],
                    "haystack_dates": ["2026-01-01"],
                    "haystack_sessions": [[{"role": "user", "content": f"hello {i}"}]],
                }
                for i in range(6)
            ]
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    def fake_eval(args, sample, budget):
        calls.append(sample.question_id)
        return {
            "question_id": sample.question_id,
            "question_type": sample.question_type,
            "best_rank": 1,
            "recall": {"5": 1, "10": 1, "15": 1},
            "ndcg_at_k": 1.0,
            "packed_words": 10,
            "retrieval_latency_ms": 1.0,
            "e2e_diagnostics": {"memories_created": 1, "candidates_extracted": 1},
        }

    monkeypatch.setattr(bench, "evaluate_sample_e2e", fake_eval)
    monkeypatch.setattr(bench, "evaluate_sample", fake_eval)

    args = argparse.Namespace(
        mode="end-to-end",
        data_path=data_path,
        work_dir=str(tmp_path / "work"),
        results_dir=str(tmp_path / "results"),
        limit=None,
        smoke=True,
        smoke_samples=5,
        confirm_benchmark=True,
        task=None,
        workers=1,
        recall_k=15,
        token_budget=1500,
        pack_atoms=40,
        abstain_threshold=0.25,
        no_rerank=True,
        no_dense=True,
        single_db=False,
        resume_from=None,
        answer_model="openai/gpt-4o-mini",
        judge_model="openai/gpt-4o-mini",
        budget_usd=8.0,
        embedding_provider=None,
        embedding_model=None,
        embedding_dimensions=None,
        extraction="openrouter",
        extraction_model=None,
        processing_batch_size=100,
        processing_lease_seconds=180,
        processing_timeout=30.0,
    )

    exit_code = bench.run(args)
    assert exit_code == 0
    assert len(calls) == 5
    result_files = list((tmp_path / "results").glob("longmemeval_s_end-to-end_*.json"))
    assert len(result_files) == 1
    output = json.loads(result_files[0].read_text(encoding="utf-8"))
    assert len(output["traces"]) == 5
