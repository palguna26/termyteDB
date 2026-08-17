# Gaps and measurable opportunities

| Dimension | Strongest observed | Remaining weakness | TermyteDB improvement and gate | Stage |
|---|---|---|---|---|
| Evidence correctness | Graphiti/Cognee provenance | Model claims can exceed source | Reject candidate without source span; >=98% attribution on labelled set | MVP |
| Temporal correctness | Graphiti invalidation | Not consistent across systems | Valid-time plus recorded-time tests; >=95% interval accuracy | MVP |
| Contradictions | Graphiti edge invalidation; Mem0 update choice | Conflict rationale/history weak | Preserve both, mark conflict, explicit supersession; >=90% classification | MVP |
| Retrieval precision | Hybrid Graphiti/Mem0 rerank | Plausible stale context | scope+validity filter before ranking; Precision@5 gate | MVP |
| Recall | Cognee/Graphiti graph expansion | Operational cost | lexical+vector parallel candidates; Recall@20 target set | MVP |
| Abstention | No strong confirmed implementation | Systems tend to return something | support threshold and empty result; calibrated abstention set | MVP |
| Isolation | Tencent scoped locks; Graphiti groups | Prompt-level isolation risk | DB predicates and adversarial leakage tests = 0 | MVP |
| Token efficiency | Tencent task context; summaries | Summaries may omit evidence | token budget, diversity, provenance compression; usefulness/token | MVP |
| Ingestion/retrieval latency | Tencent async pipeline; Mem0 simple API | model latency and provider cost | sync persist under 50 ms; async extraction; local search p95 <150 ms | MVP |
| Recovery | Tencent checkpoints/DLQ | complexity and partial state | idempotency keys, leases, retry/DLQ; kill/restart test | MVP |
| Local/hosted complexity | Mem0 local, Cognee adapters | hosted stacks multiply | SQLite MVP; Postgres production path; no graph DB initially | MVP |
| Coding continuation | Tencent specialized pipeline | framework coupling | framework-neutral events and continuation benchmark | post-MVP gate |
| Inspectability | schemas/history across systems | no unified decision trace | processing and retrieval diagnostics persisted | MVP |

Do not pursue every advantage. The first product differentiates on evidence correctness, temporal/contradiction lifecycle, isolation, inspectability, and coding-agent continuation—not on number of vector providers or graph features.

