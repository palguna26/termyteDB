# Canonical data model

All IDs are UUIDs. Every row containing memory or evidence has `org_id` and `namespace_id`; authorization filters are applied before retrieval. `created_at` is recorded time; `valid_from`/`valid_to` are asserted time. `valid_to = NULL` means currently valid, not permanently true.

```text
namespace(id, org_id, parent_id, kind, name, policy_json, deleted_at)
actor(id, org_id, kind, external_id, metadata_json)
agent(id, org_id, actor_id, name, version)
stream(id, namespace_id, actor_id, agent_id, external_id, started_at, ended_at)
event(id, namespace_id, stream_id, idempotency_key, type, occurred_at, payload_json, content_hash, redaction_state)
artifact(id, namespace_id, event_id, media_type, uri_or_text, content_hash, retention_until)
episode(id, namespace_id, stream_id, start_event_id, end_event_id, status, summary)
memory(id, namespace_id, kind, subject_key, status, confidence, current_version_id)
memory_version(id, memory_id, version, statement, structured_json, valid_from, valid_to, recorded_at, reason, model_run_id)
evidence_ref(id, memory_version_id, event_id, artifact_id, start_offset, end_offset, quote_hash, relation)
entity(id, namespace_id, canonical_key, label, resolution_confidence)
relationship(id, namespace_id, subject_entity_id, predicate, object_entity_id, status)
context_request(id, namespace_id, query, task_json, token_budget, created_at)
retrieval_candidate(id, request_id, memory_version_id, source, raw_score, rank, exclusion_reason)
context_delivery(id, request_id, selected_json, token_count, abstained, diagnostics_json)
feedback(id, namespace_id, request_id, memory_id, label, actor_id, note)
processing_job(id, namespace_id, kind, input_hash, status, attempts, lease_until, error_json, model_run_id)
```

Identity: event idempotency is `(namespace_id, idempotency_key)`; otherwise content hash plus producer ID is only a deduplication hint. Memory identity is a normalized subject/type key, never text similarity alone. A new version never destroys the old one. Statuses are `candidate`, `active`, `superseded`, `contradicted`, `invalidated`, `deleted`; deletion tombstones the logical object and removes index visibility while retaining only what policy permits.

Generated memory requires at least one evidence reference. Low-confidence or unsupported candidates are quarantined. Contradictions create a new version and a conflict relation; supersession is explicit and does not erase history. Branch/execution scope belongs in namespace metadata or stream ID.

