# Direct Memory Pipeline Design

## Goal

Make TermyteDB behave like Mem0 at the API and execution level while keeping TermyteDB's current memory model, temporal reasoning, evidence, reconciliation, retrieval, and reranking.

An ingestion call must complete extraction, embedding, reconciliation, and storage before it returns. Applications must not need to call `process()`.

## Scope

This change replaces the main queue-based ingestion workflow with a direct pipeline. It does not replace TermyteDB's memory kinds, temporal fields, evidence validation, memory history, superseding rules, search, context packing, or reranking.

## Public Behavior

- `ingest()` accepts one event and returns after its memories are stored.
- `ingest_batch()` accepts related events and handles them as one logical input.
- A batch uses one extraction-provider call unless it cannot fit within the provider's input limit.
- Extracted memory statements are embedded together.
- Validated memories and history are stored together.
- Returned results report the stored events and memory-processing outcome.
- A separate `process()` call is not required in normal use.

Compatibility helpers may remain temporarily, but they must not create or depend on pending jobs.

## Data Flow

1. Validate events, redact secrets, enforce payload limits, and verify that one batch belongs to one namespace.
2. Store raw events and artifacts with idempotency protection.
3. Build one evidence map from the new events and a bounded recent-context window.
4. Retrieve related existing memories using the current TermyteDB retrieval logic.
5. Send one extraction request containing the new evidence, recent context, and existing memory references.
6. Validate every candidate using current evidence and temporal rules.
7. Embed all accepted candidate statements with one `embed_many()` call.
8. Reconcile candidates using the current insert, reinforce, update, supersede, dispute, and ignore behavior.
9. Commit memory versions, evidence references, extraction records, and embeddings.
10. Refresh affected episode summaries and return the completed result.

## Storage

Raw events remain the source evidence. Memories, versions, evidence references, extraction runs, decisions, episodes, and vector indexes remain in place.

Processing jobs, leases, heartbeats, job claiming, and dead-letter handling are removed from the direct path. A schema migration may leave the old table present for compatibility, but new ingestion must not write jobs.

The direct pipeline should use a transaction boundary that prevents partially committed memories. Raw event storage may remain committed if the provider fails so the failed input is auditable and can be submitted again with the same idempotency keys.

## Error Handling

- Validation and idempotency conflicts fail before extraction.
- Provider and embedding errors are returned to the caller instead of becoming background job failures.
- A failed extraction does not create partial memories.
- Candidate-level validation failures are recorded and do not fail valid candidates from the same call.
- Batch fallback behavior must not silently change one logical input into unrelated extraction requests.

## Concurrency

One call runs its phases in order. Separate calls may run concurrently. Existing database locks and optimistic version checks must protect shared memory updates. There is no internal worker pool or durable processing queue in the main workflow.

## Retrieval

Search and context APIs stay unchanged. They continue to use TermyteDB's current lexical and dense retrieval, temporal and lifecycle filtering, context packing, and optional reranking.

## Benchmark

The end-to-end benchmark will submit each related conversation or session through direct batch ingestion. It will not loop over `process()`. Diagnostics will report direct extraction and memory results instead of processing-job counts.

## Tests

Tests must cover:

- single ingestion completes memory creation before returning;
- batch ingestion performs one extraction call;
- extracted statements use batch embedding;
- events remain idempotent;
- provider failure creates no partial memories;
- temporal reconciliation and evidence references remain correct;
- retrieval finds memories immediately after ingestion;
- independent concurrent calls remain safe;
- the end-to-end benchmark uses the direct API;
- legacy queue assumptions are removed or updated.

## Out of Scope

- Replacing TermyteDB's memory algorithm with Mem0's algorithm;
- changing temporal reasoning or memory kinds;
- redesigning search, context packing, or reranking;
- adding a distributed task system;
- adding automatic provider-level concurrency.
