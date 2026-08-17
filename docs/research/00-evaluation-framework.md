# Evaluation framework

This review treats repository code, tests, schemas, migrations, prompts, and configuration as evidence. README claims are recorded only when a source path confirms them. The fetched checkouts do not expose usable Git commit IDs, so conclusions are tied to the supplied snapshot and repository-relative paths.

## Comparison criteria and good performance

| Criterion | Good means |
|---|---|
| Input/event model | Accepts typed events plus opaque payloads without losing ordering or identity. |
| Raw evidence | Immutable, addressable source retained before model processing. |
| Episodes | Deterministic grouping with explicit boundaries and source links. |
| Memory types | Small typed facts, procedures, decisions, failures, and summaries with schemas. |
| Extraction | Model output is schema-validated and linked to evidence spans. |
| Deterministic/model split | Parsing, scope, idempotency, lifecycle, and authorization do not depend on a model. |
| Validation | Unsupported claims are rejected or quarantined. |
| Deduplication/consolidation | Duplicate and update decisions are explainable and reversible. |
| Version/time | History is append-only; valid-time and recorded-time are separate. |
| Contradiction/supersession | Conflicts remain inspectable and current state is explicit. |
| Entity/relationship | Resolution is scoped and confidence-aware; graph is optional, not authoritative by default. |
| Isolation/permissions | Scope predicates are enforced in storage queries and indexes. |
| Retrieval | Lexical, vector, metadata, and temporal candidates can be combined and inspected. |
| Context | Bounded, diverse, provenance-rich, and allowed to abstain. |
| Storage/deployment | One local install; hosted scale adds services only when measured necessary. |
| Jobs/recovery | Idempotent jobs, retries, checkpoints, leases, and dead letters. |
| Provider abstraction | Small interfaces for models, embeddings, storage, tokens. |
| Observability | Every decision exposes reason, versions, latency, tokens, and cost. |
| Privacy/lifecycle | Redaction, retention, export, deletion, and secret separation are testable. |
| Testing/benchmarks | Unit, integration, isolation, failure, component, retrieval, and end-to-end gates. |
| Complexity/cost | p95 latency and cost budgets are measured per operation. |

Recommended initial gates: zero cross-scope reads in adversarial tests; 100% generated memories have evidence; unsupported-memory rejection measured on a labelled set; p95 local search under 150 ms without model calls; async ingestion is recoverable after process termination; context has a configurable token cap.

