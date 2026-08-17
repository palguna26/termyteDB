# Reuse ledger

| Source | Reuse concept | TermyteDB treatment | Code/license action |
|---|---|---|---|
| Supermemory | typed API, processing state, version relations, review queue | adopt contract discipline; require evidence refs | MIT; no code copied |
| Cognee | staged tasks, provenance, rollback/orphan tests, tenant migrations | adopt pipeline/test ideas; omit backend breadth | Apache-2.0 + NOTICE; no code copied |
| Tencent | append-first events, session serialization, checkpoint, leases/DLQ | adopt recovery patterns; remove agent-specific adapters | MIT; no code copied |
| Mem0 | simple add/search/update/history, provider factories, operation classification | adopt API shape; put evidence gate before reconciliation | Apache-2.0; no code copied |
| Graphiti | episodes, temporal invalidation, group filters, hybrid candidate retrieval | adopt data semantics; defer graph DB | Apache-2.0; no code copied |

Attribution is recorded now because future implementation may be influenced by these public patterns. No source code, prompt text, or asset is copied into this repository.

