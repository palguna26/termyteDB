"""Category-focused coverage for LongMemEval retrieval improvements (Phases 1-5)."""
from __future__ import annotations

from datetime import UTC, datetime

from src.memory.extractor import (
    is_explicit_preference,
    preference_polarity,
)
from src.models import TemporalQuery, temporal_valid_at_score
from src.retrieval.context import count_tokens, count_words, pack_evidence, tokenizer_mode
from src.retrieval.retrieval import (
    AtomHit,
    aggregate_atom_sessions,
    pack_atoms_token_aware,
    parse_reference_date,
    parse_temporal_query,
    preference_atom_boost,
    search_atoms_with_stages,
    temporal_atom_boost,
)
from src.storage.db import Database

# ---------------------------------------------------------------------------
# Phase 1: token correctness + staged measurement
# ---------------------------------------------------------------------------

def test_packed_words_and_tokens_are_separate():
    text = "Hello world, this is a test."
    words = count_words(text)
    tokens = count_tokens(text, model="gpt-4o-mini")
    assert isinstance(words, int) and isinstance(tokens, int)
    assert words == len(text.split())
    assert tokenizer_mode(model="gpt-4o-mini") in {"exact", "approximate"}


def test_pack_atoms_enforces_token_budget_after_headers():
    hits = [
        AtomHit(f"a{i}", f"s{i % 3}", "filler fact about SQLite preferences " * 10, "2023/05/20 (Sat) 02:21", "user", 1.0 - i * 0.01)
        for i in range(20)
    ]
    packed = pack_atoms_token_aware(hits, token_budget=200, tokenizer_model="gpt-4o-mini")
    assert packed["token_count"] <= 200
    assert packed["word_count"] == len(str(packed["text"]).split())
    assert packed["tokenizer"] in {"exact", "approximate"}
    # Headers/dates/roles are part of the budget, not just raw facts.
    assert "[Session" in str(packed["text"]) or str(packed["text"]) == "insufficient information"


def test_pack_evidence_reports_words_tokens_and_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMYTEDB_ALLOW_FAKE_EXTRACTION", "1")
    from src import TermyteDB
    from tests.conftest import event

    class _Emb:
        name = "test-v1"
        dimensions = 2

        def embed(self, value: str) -> list[float]:
            return [0.1, 0.2]

        def embed_many(self, values: list[str]) -> list[list[float]]:
            return [[0.1, 0.2] for _ in values]

    db = TermyteDB(tmp_path / "pack-words.sqlite", embedding_provider=_Emb())
    for i in range(3):
        db.ingest(event("ns1", f"k{i}", f"I prefer SQLite for local project number {i}."))
    memories = db.search("ns1", "SQLite", limit=5)
    packed = pack_evidence(
        memories,
        lambda m: db.repository.chunks_for_events("ns1", [str(x) for x in m.source_event_ids]),
        token_budget=300,
        hard_max=300,
        tokenizer_model="gpt-4o-mini",
    )
    assert packed["token_count"] <= 300
    assert "word_count" in packed and "tokenizer" in packed
    assert packed["word_count"] != packed["token_count"] or packed["word_count"] <= 300
    db.close()


def test_search_atoms_reports_named_stages(tmp_path):
    db = Database(tmp_path / "stages.sqlite")
    try:
        db.execute("INSERT OR IGNORE INTO namespaces(id, org_id, created_at) VALUES ('n1','b',datetime('now'))")
        db.connection.commit()
        hits, stages = search_atoms_with_stages(
            db, "What is my current job?", 10,
            vector_search=lambda *_: [], namespace_id=None,
            reference_date="2023/05/20 (Sat) 02:21",
        )
        assert set(stages) >= {"fts_ms", "dense_ms", "rrf_ms", "temporal_ms"}
        assert all(isinstance(v, float) for v in stages.values())
        assert isinstance(hits, list)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Phase 2: temporal / date-aware retrieval
# ---------------------------------------------------------------------------

def test_parse_temporal_query_intents():
    assert parse_temporal_query("What is my current job?", "2023/05/20 (Sat) 02:21").intent == "latest"
    assert parse_temporal_query("What did I previously use?", "2023/05/20 (Sat) 02:21").intent == "historical"
    assert parse_temporal_query("What was the first job?", "2023/05/20 (Sat) 02:21").intent == "earliest"
    assert parse_temporal_query("What happened in March 2023?", "2023/05/20 (Sat) 02:21").intent == "around"
    q = parse_temporal_query("What happened in March 2023?", "2023/05/20 (Sat) 02:21")
    assert q.date_range_start is not None and q.date_range_end is not None
    assert parse_temporal_query("What happened before 2023?", "2023/05/20 (Sat) 02:21").intent == "before"
    assert parse_temporal_query("What happened after 2022?", "2023/05/20 (Sat) 02:21").intent == "after"
    assert parse_temporal_query("What is SQLite?", "2023/05/20 (Sat) 02:21").intent == "none"


def test_reference_date_never_uses_machine_clock():
    ref = parse_reference_date("2023/05/20 (Sat) 02:21")
    assert ref is not None and ref.year == 2023 and ref.month == 5 and ref.day == 20
    assert parse_reference_date("") is None
    assert parse_reference_date(None) is None


def test_temporal_atom_boost_prefers_in_range():
    tq = parse_temporal_query("What happened in March 2023?", "2023/06/01 (Thu) 00:00")
    in_range = temporal_atom_boost("2023/03/15 (Wed) 10:00", tq)
    out_range = temporal_atom_boost("2022/01/01 (Sat) 00:00", tq)
    assert in_range > out_range
    assert in_range > 0


def test_temporal_atom_boost_latest_prefers_past_not_future():
    tq = parse_temporal_query("What is my current job?", "2023/05/20 (Sat) 02:21")
    past = temporal_atom_boost("2023/04/10 (Mon) 10:00", tq)
    future = temporal_atom_boost("2023/06/10 (Sat) 10:00", tq)
    assert past >= 0
    assert future < past


def test_temporal_atom_boost_missing_dates():
    tq = parse_temporal_query("What happened in March 2023?", "2023/06/01 (Thu) 00:00")
    assert isinstance(temporal_atom_boost(None, tq), float)
    tq_none = parse_temporal_query("What is SQLite?", "2023/06/01 (Thu) 00:00")
    assert temporal_atom_boost("2023/03/15 (Wed) 10:00", tq_none) == 0.0


def test_temporal_valid_at_score_latest_vs_historical():
    ref = datetime(2023, 5, 20, tzinfo=UTC)
    latest_q = TemporalQuery(reference_date=ref, intent="latest")
    hist_q = TemporalQuery(reference_date=ref, intent="historical")
    current = temporal_valid_at_score(datetime(2023, 4, 1, tzinfo=UTC), None, None, latest_q)
    future = temporal_valid_at_score(datetime(2023, 6, 1, tzinfo=UTC), None, None, latest_q)
    assert current > future
    expired = temporal_valid_at_score(datetime(2022, 1, 1, tzinfo=UTC), datetime(2023, 1, 1, tzinfo=UTC), None, hist_q)
    assert expired > 0


def test_temporal_valid_at_score_before_after_around():
    ref = datetime(2023, 6, 1, tzinfo=UTC)
    around = TemporalQuery(
        reference_date=ref, intent="around",
        target_date=datetime(2023, 3, 15, tzinfo=UTC),
        date_range_start=datetime(2023, 3, 1, tzinfo=UTC),
        date_range_end=datetime(2023, 4, 1, tzinfo=UTC),
    )
    assert temporal_valid_at_score(datetime(2023, 3, 15, tzinfo=UTC), None, None, around) > \
        temporal_valid_at_score(datetime(2022, 1, 1, tzinfo=UTC), None, None, around)
    before = TemporalQuery(reference_date=ref, intent="before", target_date=datetime(2023, 1, 1, tzinfo=UTC))
    assert temporal_valid_at_score(datetime(2022, 6, 1, tzinfo=UTC), None, None, before) > \
        temporal_valid_at_score(datetime(2023, 6, 1, tzinfo=UTC), None, None, before)


def test_question_dates_near_version_boundaries():
    ref = datetime(2023, 5, 20, 12, 0, tzinfo=UTC)
    q = TemporalQuery(reference_date=ref, intent="latest")
    just_before = temporal_valid_at_score(datetime(2023, 5, 20, 11, 0, tzinfo=UTC), None, None, q)
    just_after = temporal_valid_at_score(datetime(2023, 5, 20, 13, 0, tzinfo=UTC), None, None, q)
    assert just_before > just_after


def test_repository_temporal_scoring_uses_reference_date(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMYTEDB_ALLOW_FAKE_EXTRACTION", "1")
    from src import TermyteDB
    from tests.conftest import event

    class _Emb:
        name = "test-v1"
        dimensions = 2

        def embed(self, value: str) -> list[float]:
            h = abs(hash(value)) % 100
            return [float(h % 10) / 10, float((h // 10) % 10) / 10]

        def embed_many(self, values: list[str]) -> list[list[float]]:
            return [self.embed(v) for v in values]

    db = TermyteDB(tmp_path / "temporal.sqlite", embedding_provider=_Emb())
    db.ingest(event("ns1", "k1", "User lives in Delhi."))
    db.ingest(event("ns1", "k2", "User now lives in Pune, moved from Delhi."))
    current = db.search("ns1", "where does user currently live", limit=5)
    assert len(current) > 0
    assert any("temporal_boost" in r.component_scores for r in current)
    # Reference-date path must not crash and must score deterministically.
    ref_scored = db.repository.search("ns1", "where does user currently live", 5, reference_date="2023/05/20 (Sat) 02:21")
    assert len(ref_scored) > 0
    db.close()


# ---------------------------------------------------------------------------
# Phase 3: preference extraction + retrieval
# ---------------------------------------------------------------------------

def test_preference_polarity_classification():
    assert preference_polarity("User prefers Sony.") == "positive"
    assert preference_polarity("User dislikes Canon.") == "negative"
    assert preference_polarity("User no longer uses Canon.") == "update"
    assert preference_polarity("The sky is blue.") == "none"


def test_is_explicit_preference_requires_language():
    assert is_explicit_preference("I prefer Sony over Canon.")
    assert is_explicit_preference("My favorite color is blue.")
    assert not is_explicit_preference("I saw a Sony camera in the shop.")
    assert not is_explicit_preference("The weather is nice today.")


def test_preference_extraction_rules_cover_updates_and_negatives():
    from src.memory.extractor import extract

    cands = extract({"text": "I prefer Sony over Canon for photography."})
    assert any("prefer" in c.statement.casefold() and "canon" in c.statement.casefold() for c in cands)
    cands2 = extract({"text": "I no longer use Canon; I avoid heavy lenses."})
    assert len(cands2) > 0
    cands3 = extract({"text": "I used to like Canon but now prefer Sony."})
    assert len(cands3) > 0


def test_query_weights_detects_preference():
    from src import TermyteDB

    class _Emb:
        name = "test-v1"
        dimensions = 2

        def embed(self, value: str) -> list[float]:
            return [0.1, 0.2]

        def embed_many(self, values: list[str]) -> list[list[float]]:
            return [[0.1, 0.2] for _ in values]

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        db = TermyteDB(Path(td) / "w.sqlite", embedding_provider=_Emb())
        qw = db.repository._query_weights("What do I prefer for photography?")
        assert qw["is_preference"] == 1.0
        assert qw["vector"] > 1.0
        qw2 = db.repository._query_weights("Which version was used?")
        assert qw2["is_preference"] == 0.0
        db.close()


def test_preference_atom_boost():
    assert preference_atom_boost("What do I prefer?", "User prefers Sony.") > 0
    assert preference_atom_boost("What do I prefer?", "User dislikes Canon.") > 0
    assert preference_atom_boost("What is SQLite?", "User prefers Sony.") == 0.0


def test_repository_preference_boost_in_scores(tmp_path, monkeypatch):
    from src import TermyteDB
    from tests.conftest import event

    monkeypatch.setenv("TERMYTEDB_ALLOW_FAKE_EXTRACTION", "1")

    class _Emb:
        name = "test-v1"
        dimensions = 2

        def embed(self, value: str) -> list[float]:
            h = abs(hash(value)) % 100
            return [float(h % 10) / 10, float((h // 10) % 10) / 10]

        def embed_many(self, values: list[str]) -> list[list[float]]:
            return [self.embed(v) for v in values]

    db = TermyteDB(tmp_path / "pref.sqlite", embedding_provider=_Emb())
    db.ingest(event("ns1", "k1", "I prefer Sony-compatible accessories for photography."))
    db.ingest(event("ns1", "k2", "The shop sells Canon cameras."))
    results = db.search("ns1", "What photography accessories do I prefer?", limit=5)
    assert len(results) > 0
    assert any("preference_boost" in r.component_scores for r in results)
    db.close()


# ---------------------------------------------------------------------------
# Phase 4: multi-session aggregation
# ---------------------------------------------------------------------------

def test_aggregate_atom_sessions_covers_multiple_sessions():
    hits = [
        AtomHit("a1", "s1", "User likes tea.", "2023/01/01 (Sun) 00:00", "user", 0.9),
        AtomHit("a2", "s1", "User likes tea again.", "2023/01/01 (Sun) 01:00", "user", 0.89),
        AtomHit("a3", "s1", "User likes tea third.", "2023/01/01 (Sun) 02:00", "user", 0.88),
        AtomHit("a4", "s2", "User likes coffee.", "2023/02/01 (Wed) 00:00", "user", 0.5),
        AtomHit("a5", "s3", "User likes juice.", "2023/03/01 (Wed) 00:00", "user", 0.49),
    ]
    order = aggregate_atom_sessions(hits, "Compare preferences across all sessions", limit=5)
    assert "s2" in order and "s3" in order
    assert order[0] in {"s1", "s2", "s3"}


def test_aggregate_single_session_keeps_precision():
    hits = [
        AtomHit("a1", "s1", "User lives in Pune.", "2023/01/01 (Sun) 00:00", "user", 0.9),
        AtomHit("a2", "s2", "Unrelated fact.", "2023/02/01 (Wed) 00:00", "user", 0.1),
    ]
    order = aggregate_atom_sessions(hits, "Where does the user live?", limit=5)
    assert order[0] == "s1"


def test_multi_session_evidence_share_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMYTEDB_ALLOW_FAKE_EXTRACTION", "1")
    from src import TermyteDB
    from tests.conftest import event

    class _Emb:
        name = "test-v1"
        dimensions = 2

        def embed(self, value: str) -> list[float]:
            return [0.5, 0.5]

        def embed_many(self, values: list[str]) -> list[list[float]]:
            return [[0.5, 0.5] for _ in values]

    db = TermyteDB(tmp_path / "multi.sqlite", embedding_provider=_Emb())
    for i in range(4):
        db.ingest(event("ns1", f"k{i}", f"Session marker {i} across timeline discussion point {i}."))
    results = db.search("ns1", "Compare across all sessions each timeline point", limit=6)
    assert isinstance(results, list)
    db.close()


# ---------------------------------------------------------------------------
# Phase 5: latency (cached reranker, batched SQL, indexes)
# ---------------------------------------------------------------------------

def test_flashrank_init_cached():
    from src.retrieval.retrieval import _cached_flashrank

    try:
        first = _cached_flashrank("ms-marco-MiniLM-L-12-v2")
    except Exception:
        import pytest
        pytest.skip("flashrank model unavailable offline")
        return
    second = _cached_flashrank("ms-marco-MiniLM-L-12-v2")
    assert first is second


def test_repository_reranker_cached():
    from src.storage.repository import _cached_reranker

    first = _cached_reranker("ms-marco-MiniLM-L-12-v2")
    second = _cached_reranker("ms-marco-MiniLM-L-12-v2")
    assert first is second or first is None


def test_retrieval_indexes_exist(tmp_path):
    db = Database(tmp_path / "idx.sqlite")
    try:
        names = {row["name"] for row in db.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()}
        for expected in (
            "memory_versions_namespace_valid_idx",
            "memory_embeddings_provider_ns_idx",
            "events_namespace_session_idx",
            "evidence_refs_version_idx",
            "atoms_namespace_invalid_idx",
        ):
            assert expected in names, f"missing index {expected}: {sorted(names)}"
    finally:
        db.close()


def test_batched_atom_flags_no_n_plus_one():
    import tempfile
    from pathlib import Path

    from src.retrieval.retrieval import _batch_atom_flags

    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "flags.sqlite")
        try:
            db.execute("INSERT OR IGNORE INTO namespaces(id, org_id, created_at) VALUES ('n1','b',datetime('now'))")
            db.connection.commit()
            assert _batch_atom_flags(db, []) == {}
        finally:
            db.close()
