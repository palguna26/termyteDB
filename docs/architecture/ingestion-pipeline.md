# Ingestion and processing pipeline

```text
ingest -> validate -> redact -> deduplicate -> commit immutable evidence
       -> episode boundary -> extract candidates -> evidence validation
       -> reconcile -> update FTS/vector indexes
```

Ingest is synchronous and deterministic through the evidence commit. Redaction, schema checks, scope checks, hashes, and idempotency are deterministic. Episode construction may be synchronous for explicit boundaries and asynchronous for inactivity windows. Extraction, entity resolution, and reconciliation are asynchronous and model-based where needed. Index updates are asynchronous but retryable.

The transaction commits the event before enqueueing a processing job. Jobs carry input hashes and are safe to repeat. A worker leases a job, records model/provider/prompt versions, writes candidates, and commits reconciliation atomically. Failed jobs retry with bounded exponential backoff, then enter a dead-letter state. A reprocessor can replay evidence without duplicating memories.

Redaction must run before external model calls; secrets are never sent to providers by default. Raw evidence remains encrypted/permissioned storage. Diagnostics record each stage, duration, token counts, provider, and failure.

