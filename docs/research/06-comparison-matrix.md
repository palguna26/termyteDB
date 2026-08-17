# Cross-system comparison

Legend: **Y** = confirmed in inspected source; **P** = partial or provider/configuration dependent; **U** = not verified in the supplied checkout; **N** = not central.

| Capability | Supermemory | Cognee | Tencent | Mem0 | Graphiti | TermyteDB decision |
|---|---:|---:|---:|---:|---:|---|
| Typed public API | Y | Y | Y | Y | Y | Small stable engine/service API |
| Immutable raw evidence | U | P | Y for pending files | P | P via episodes | Authoritative event/artifact tables |
| Evidence-linked memory | P | Y provenance | P | P history | Y episode edges | Mandatory evidence refs and spans |
| Model extraction | P/U | Y | Y | Y | Y | Async, schema-constrained |
| Deterministic validation | P | Y pipeline | Y schemas/locks | P | P | Scope, schema, hash, evidence checks |
| Duplicate/update logic | P relations | Y task/dedup tests | P pipeline state | Y ADD/UPDATE/DELETE | Y node/edge dedup | Append versions plus explicit decision |
| Temporal validity | U | P | P task timestamps | P | Y | First-class valid_from/to |
| Contradiction/supersession | U | P | U | P delete/update | P invalidation | Preserve conflict; choose current view |
| Graph representation | P memory graph | Y | P code graph | P optional graph | Y core | Relational optional index in MVP |
| Namespace isolation | P container tags | Y tenant/user/dataset | Y instance/team/agent/session | Y filters | Y group_id | Storage predicates, never prompt-only |
| Lexical retrieval | U | P | U | P BM25 | Y full text | SQLite FTS5/Postgres FTS |
| Vector retrieval | P | Y | Y | Y | Y | Optional embedding index |
| Graph retrieval | U | Y | P | P | Y | Post-MVP/experiment |
| Temporal retrieval | U | P | P | P | Y | SQL validity filters |
| Reranking/context | P rerank flag | P retrievers | P task context | P rerank/explain | Y hybrid/RRF | Explainable score + bounded context |
| Abstention | U | U | P lifecycle | U | U | Explicit low-support abstention |
| Background recovery | U | Y rollback/sync | Y queue/checkpoint/DLQ | P async | P bulk operations | Durable local jobs, hosted queue later |
| Provider abstraction | P | Y | P | Y | Y | Only required provider seams |
| Local deployment | U | Y many choices | P files/local backend | Y SQLite/vector choices | P Kuzu | SQLite first |
| Hosted scale | P | Y | Y | Y | Y | Postgres + object storage + queue only when gated |
| Coding-agent fit | P integration | P general | Y | P procedural mode | P | Primary benchmark target |
| License | MIT | Apache-2.0 | MIT | Apache-2.0 | Apache-2.0 | Avoid code copying; preserve notices |

Common pattern: all rely on model extraction plus an index. The key disagreement is authoritative storage: Mem0 is vector/history centric, Graphiti graph centric, Cognee pipeline/graph centric, Tencent event/pipeline centric, and Supermemory API/document centric. Shared gap: none fully proves evidence-span correctness, safe contradiction handling, storage-level authorization, and abstention together.

