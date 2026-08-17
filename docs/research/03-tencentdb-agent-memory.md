# TencentDB Agent Memory source review

## Confirmed implementation

This repository is unusually relevant to coding-agent continuation. `MemoryCore/src/utils/pipeline-manager.ts` documents and implements L0 buffering, L1 batch processing, L2 scene extraction, and L3 persona generation. It uses per-session state, timers, warm-up thresholds, serial queues, checkpoint persistence, shutdown flush, retry limits, and pending-session recovery. `MemoryCore/src/services/pipeline-worker.ts` adds queue consumption, lock granularity, retries, dead letters, stale pending-message claiming, and idempotent upsert/overwrite assumptions.

`MemoryCore/src/offload_server/ingest-handler.ts` validates `IngestRequestSchema`, appends tool pairs to session-scoped `pending.jsonl`, writes recent context, uses an in-process mutex plus optional distributed lock, triggers tasks by threshold/timer, and returns 409 on lock failure. This is an evidence-preserving raw event path, but it is specialized to sessions, tool pairs, COS-style files, and the repository's agent integrations.

The SDK exposes versioned clients under `sdk/memory-core/typescript/src/v3/` and Python equivalents. `MemoryKnowledge/src/db/schema.ts` contains SQLite knowledge tables, audit tables, code graph/wiki records, and LLM bindings. `MemoryCore/src/offload_server/prompts/l15-prompt.ts` uses model judgment to classify task completion, long-task status, and continuation against Mermaid task files. The repository also contains direct agent adapters (`MemoryProxy/src/agent-adapters/`), so it does not satisfy TermyteDB's framework-independent boundary by itself.

## Engineering assessment

Strong decisions: append-before-process ingestion; per-session serialization; explicit lock failure; checkpoint/recovery; bounded retries and dead letters; warm-up and timer semantics; coding-task state representation.

Weaknesses: integration-specific adapters and prompts; file/COS and queue assumptions; model judgment of task lifecycle; no demonstrated general contradiction/reconciliation model; broad system coupling between proxy, panel, knowledge, and core.

## Reuse and rejection

Reuse the event-first write path, idempotent jobs, checkpoint, lease/recovery, and dead-letter patterns. Reject direct agent-adapter logic, Mermaid-specific state, and Redis/queue requirements for local V1. License is MIT (`LICENSE`); retain notice if code is reused.

