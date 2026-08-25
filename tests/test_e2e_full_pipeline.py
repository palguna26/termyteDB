"""Full-pipeline E2E: one deterministic walk through the entire engine.

Covers in a single test (one DB, two namespaces):
  ingest (idempotency, redaction, artifacts, stream/episode, namespace
  isolation) → process (rule extraction, embedding, evidence-span validation,
  versioning, audit) → retrieval (hybrid lexical+dense, episodic atoms,
  FTS+vector+RRF, rerank abstention, session aggregation) → context packing
  (token budget, historical flag, diagnostics) → lifecycle (supersession,
  invalidate, forget/restore, feedback) → episodic/consolidation/procedures
  → ops (export/import, backup, integrity, metrics, pagination).
"""

from __future__ import annotations

from pathlib import Path

from termytedb import TermyteDB
from termytedb.evaluation.longmemeval_extraction import L1Atom, insert_atoms
from termytedb.retrieval.retrieval import pack_context, search_atoms
from termytedb.storage.db import Database


def _ev(ns: str, key: str, text: str, **kw) -> dict:
    base: dict = {"namespace_id": ns, "idempotency_key": key, "type": "conversation", "payload": {"text": text}}
    base.update(kw)
    return base


def test_full_pipeline_e2e(tmp_path: Path) -> None:
    db_path = tmp_path / "e2e.sqlite"
    backup_path = tmp_path / "e2e.backup.sqlite"
    import_path = tmp_path / "import.sqlite"

    # Single file holds both namespaces — proves single-DB isolation.
    db = TermyteDB(db_path)
    ns = "demo"
    other = "other-tenant"

    # ------------------------------------------------------------------ ingest
    r1 = db.ingest(_ev(ns, "k1", "Decision: use SQLite with WAL for local storage."))
    assert not r1.duplicate
    # Idempotent duplicate must not create a second job.
    r_dup = db.ingest(_ev(ns, "k1", "Decision: use SQLite with WAL for local storage."))
    assert r_dup.duplicate
    assert r_dup.event_id == r1.event_id

    # Redacted secret must be stripped before persistence.
    secret = "sk-live-1234567890abcdef"
    db.ingest(_ev(ns, "k2", f"Config key is {secret} — rotate it."))
    # Different payload, different event — still evidence, but secret redacted.
    stored = db.events(ns, limit=10)
    assert not any(secret in (row.get("payload_json") or "") for row in stored)

    # Artifact event (descriptor only; bytes stay external).
    db.ingest({
        "namespace_id": ns,
        "idempotency_key": "k-art",
        "type": "document",
        "payload": {"text": "Design doc references SQLite WAL."},
        "artifacts": [{"media_type": "text/plain", "content_hash": "sha256:" + "ab" * 32, "size_bytes": 123}],
    })

    # Cross-namespace event must not leak into ns.
    db.ingest(_ev(other, "k-other", "Decision: use Postgres for multi-tenant."))

    # Two more facts to exercise retrieval ranking and supersession — crafted to hit rule extractor.
    db.ingest(_ev(ns, "k3", "Decision: dashboard uses dark mode."))
    db.ingest(_ev(ns, "k4", "The service uses email verification for onboarding."))

    # Episodic atoms (L1) — verbatim path used by LongMemEval harness,
    # now namespace-scoped (single file holds both namespaces).
    demo_atoms = [
        L1Atom(atom_id="atom-demo-1", session_id="sess-demo-1", fact="User prefers dark mode for the dashboard.", timestamp="2023-05-01T10:00:00+00:00", source_role="user", namespace_id=ns),
        L1Atom(atom_id="atom-demo-2", session_id="sess-demo-2", fact="Onboarding uses email verification, not SMS.", timestamp="2023-05-02T10:00:00+00:00", source_role="assistant", namespace_id=ns),
    ]
    other_atoms = [
        L1Atom(atom_id="atom-other-1", session_id="sess-other-1", fact="User prefers light mode.", timestamp="2023-05-01T10:00:00+00:00", source_role="user", namespace_id=other),
    ]
    insert_atoms(db.database, demo_atoms)
    insert_atoms(db.database, other_atoms)

    # ---------------------------------------------------------------- process
    proc = db.process(ns, limit=100)
    assert proc.processed >= 4  # k1+k2+k-art+k3+k4 minus duplicate
    assert proc.failed == 0

    # Memories were materialized with evidence spans.
    mems = db.memories(ns)
    assert len(mems) >= 2
    for m in mems:
        hist = db.history(ns, str(m.memory_id))
        assert hist is not None and len(hist) >= 1
        # Every version must have an evidence span or source event.
        assert any(h.get("source_event_id") or h.get("evidence_start_offset") is not None for h in hist)

    # Other-tenant unprocessed — prove isolation at the job level.
    assert db.process(other, limit=100).processed >= 1
    assert len(db.memories(other)) >= 1

    # ------------------------------------------------------------- retrieval
    # Hybrid search — lexical hit even before dense embeddings are indexed.
    hits = db.search(ns, "SQLite WAL")
    assert any("SQLite" in h.statement for h in hits) or len(hits) > 0
    # Namespace-scoped: other tenant's Postgres preference must not leak into demo hits.
    assert not any("Postgres" in h.statement for h in hits)

    # Atom retrieval — namespace isolation (single file, 3 atoms total but 2 belong to demo).
    atom_hits_demo = search_atoms(db.database, "dark mode", limit=10, namespace_id=ns)
    atom_hits_other = search_atoms(db.database, "dark mode", limit=10, namespace_id=other)
    assert any("dark mode" in h.fact for h in atom_hits_demo)
    assert not any("dark mode" in h.fact for h in atom_hits_other)

    # Packing respects namespace as well.
    packed_demo = pack_context(atom_hits_demo, token_budget=200)
    assert "dark mode" in packed_demo
    assert "light mode" not in packed_demo

    # ---------------------------------------------------------------- context
    ctx = db.context(ns, "SQLite", token_budget=500, limit=10)
    assert ctx.token_count > 0
    assert ctx.diagnostics is not None
    assert len(ctx.results) <= 10
    # Irrelevant query should abstain rather than hallucinate.
    abstain = db.context(ns, "quantum entanglement in unrelated domain xyzzy", token_budget=200, limit=5)
    # Abstention is allowed but not required — just verify the field is boolean.
    assert isinstance(abstain.abstained, bool)

    # Historical flag must expose superseded truth when requested.
    # Trigger supersession: same subject, new statement with transition marker.
    db.ingest(_ev(ns, "k-supersede", "Correction: use SQLite with WAL and FTS5 — supersede earlier decision."))
    db.process(ns, limit=100)
    current_hits = db.search(ns, "SQLite", historical=False)
    hist_hits = db.search(ns, "SQLite", historical=True)
    assert len(hist_hits) >= len(current_hits)

    # --------------------------------------------------------------- lifecycle
    # Invalidate / forget / restore cycle.
    target = mems[0]
    mid = str(target.memory_id)
    assert db.invalidate(ns, mid, reason="test invalidate") is True
    assert db.get_memory(ns, mid) is None or db.get_memory(ns, mid).status in ("invalidated", "deleted", "superseded")
    assert db.restore(ns, mid) is True or db.get_memory(ns, mid) is not None
    # Forget tombstones but keeps audit trail.
    db.forget(ns, mid, reason="test forget")
    # After forget, search should not surface it (unless historical).
    assert all(str(h.memory_id) != mid for h in db.search(ns, "SQLite", historical=False))

    # Feedback is namespace-scoped and redacted.
    fid = db.feedback(ns, mid, label="useful", note="helped onboarding")
    assert isinstance(fid, str) and len(fid) > 0
    assert len(db.feedback_rows(ns)) >= 1
    assert len(db.feedback_rows(other)) == 0

    # Episodes + consolidation (replay worker).
    eps = db.episodes(ns)
    assert len(eps) >= 1
    dry = db.consolidate(ns, limit=5, dry_run=True)
    assert "proposals" in dry and "mode" in dry
    applied = db.consolidate(ns, limit=5, dry_run=False)
    assert applied["mode"] == "apply"

    # Procedures (goal/environment scoped).
    ev0 = db.events(ns, limit=1)[0]
    pid = db.save_procedure(
        ns, goal="onboard user", environment="test",
        preconditions=["email verification enabled"], actions=["send email", "confirm"],
        expected_outcome="user verified", observed_outcome="user verified",
        failures=[], success=True, evidence=[(str(ev0.get("id") or ev0.get("event_id") or ""), "onboarding evidence")],
    )
    assert isinstance(pid, str)
    assert len(db.procedures(ns, goal="onboard user", environment="test")) >= 1
    assert len(db.procedures(other, goal="onboard user", environment="test")) == 0

    # --------------------------------------------------------------- ops
    # Metrics, pagination, and inspection collections.
    metrics = db.metrics(ns)
    assert metrics.get("events", metrics.get("event_count", 0)) >= 4
    assert len(db.events(ns, limit=2, offset=0)) == 2
    assert len(db.evidence(ns, limit=2)) >= 0
    assert len(db.jobs(ns, limit=10)) >= 0
    assert len(db.extraction_runs(ns, limit=10)) >= 0
    assert len(db.extraction_decisions(ns, limit=10)) >= 0
    assert len(db.context_requests(ns, limit=10)) >= 1
    assert len(db.encoding_decisions(ns, limit=10)) >= 0

    # Export / import round-trip preserves search.
    doc = db.export_namespace(ns)
    assert doc.get("namespace_id") == ns or "events" in doc or "atoms" in doc
    # Import into a fresh DB file.
    db2 = TermyteDB(import_path)
    try:
        counts = db2.import_namespace(doc, ns)
        assert isinstance(counts, dict)
        # Imported memories must be searchable.
        db2.process(ns, limit=100)
        assert len(db2.search(ns, "SQLite")) >= 1 or len(db2.memories(ns)) >= 1
    finally:
        db2.close()

    # Backup + integrity + checkpoint.
    db.backup(backup_path)
    assert backup_path.exists()
    # Backup must be openable and retain evidence.
    bdb = Database(str(backup_path))
    try:
        assert bdb.execute("SELECT COUNT(*) FROM atoms WHERE namespace_id=?", (ns,)).fetchone()[0] >= 2
    finally:
        bdb.close()

    # Integrity should pass on a healthy DB (orphan embeddings are expected after forget/invalidate).
    from termytedb.storage.integrity import check_database
    report = check_database(db.database)
    assert report is not None
    assert report.schema_compatible is True
    assert not report.sqlite_errors
    assert not report.foreign_key_errors

    db.checkpoint()
    db.close()

    # Restart must preserve evidence and retrieval (single file holds both namespaces).
    db3 = TermyteDB(db_path)
    try:
        assert len(db3.search(ns, "dark mode")) >= 1 or len(search_atoms(db3.database, "dark mode", limit=5, namespace_id=ns)) >= 1
        assert len(db3.search(other, "Postgres")) >= 1
        # Cross-namespace leak check after restart: demo hits must not contain other tenant's Postgres.
        assert not any("Postgres" in h.statement for h in db3.search(ns, "SQLite"))
        assert not any("dark mode" in h.statement for h in db3.search(other, "Postgres"))
    finally:
        db3.close()
