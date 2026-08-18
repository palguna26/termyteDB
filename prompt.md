Build TermyteDB completely into a shippable, evidence-backed, benchmarked V1 by autonomously planning, implementing, testing, validating, profiling, documenting, and repairing the system milestone by milestone.

Treat the repository’s existing research, architecture documents, ADRs, build plan, risk register, tests, and completed Milestones 1 and 1.1 as the authoritative starting point. Inspect the actual repository and Git history before changing anything. Preserve every established invariant unless concrete evidence requires a change; document material changes in ADRs.

TermyteDB is a framework-independent memory engine for AI agents. It must ingest agent activity and other evidence, convert that evidence into structured and temporally correct memories, reconcile new information with existing knowledge, retrieve task-relevant memories, and return compact context with verifiable provenance. TermyteDB must not contain Codex-, Claude Code-, Cursor-, LangGraph-, Termyte CLI-, or other vendor-specific integration logic.

Continue working without waiting for routine confirmation. After every meaningful implementation step, run the relevant validation, inspect actual failures, repair underlying causes, rerun validation, and continue. Do not mark the Goal complete based on code appearance or narrated confidence. Completion requires concrete test, evaluation, benchmark, integrity, security, restart, and Git evidence.

## Required V1 capabilities

Build all of the following:

### 1. Evidence ingestion

* Versioned canonical event protocol.
* Immutable evidence storage.
* Namespace, stream, actor, agent, session, and source identity.
* Namespace-scoped idempotency.
* Deterministic canonical hashing.
* Batch ingestion.
* Safe normalization.
* Deterministic redaction before persistence.
* Artifact metadata and content-addressed references where justified.
* Idempotent retries.
* Explicit conflict behavior.
* Size and rate boundaries.
* Append-first processing.
* Import and export of canonical events.
* Complete ingestion diagnostics.

### 2. Episode construction

* Group related events into coherent task episodes.
* Preserve event order and chronology.
* Use deterministic boundaries first.
* Use a model only for genuinely ambiguous boundaries.
* Support active, completed, failed, abandoned, and interrupted episodes.
* Represent goals, attempts, discoveries, decisions, outcomes, failures, verification, and unfinished work.
* Handle long sessions, compaction, late events, out-of-order events, restarts, and duplicated capture.
* Keep episode construction independently testable and replaceable.

### 3. Memory extraction

* Strict versioned structured-output schema.
* Memory types including facts, decisions, attempts, failures, outcomes, constraints, procedures, task state, corrections, and unresolved questions.
* Minimal model-provider boundary.
* Deterministic fake provider for tests.
* At least one real configurable extraction provider.
* Local or existing-agent provider support only if it can be implemented cleanly.
* Bounded extraction batches.
* Prompt-injection-resistant evidence framing.
* Exact persisted evidence spans.
* Deterministic evidence-span verification.
* Conservative semantic-support verification.
* Candidate normalization and fingerprinting.
* Rejection of unsupported, secret-containing, malformed, duplicated, cross-namespace, oversized, or weak claims.
* No direct model writes to authoritative memory tables.
* Rule-only operation when model extraction is disabled.

### 4. Memory reconciliation

Implement auditable:

* INSERT
* REINFORCE
* UPDATE
* SUPERSEDE
* DISPUTE
* IGNORE

Support:

* Logical memory identity.
* Append-only memory versions.
* Exact evidence references.
* Valid-from and valid-until intervals.
* Current and historical truth.
* Corrections.
* Contradictions.
* Superseded knowledge.
* Disputed knowledge.
* Late evidence.
* Out-of-order processing.
* Duplicate extraction.
* Concurrent reconciliation.
* Confidence and importance.
* Temporary versus durable memory.
* Explicit invalidation.
* Full decision audit trail.

Recency alone must never establish truth. A model-proposed action must never bypass deterministic validation.

### 5. Indexing and retrieval

Build an explainable hybrid retrieval pipeline using:

* FTS5 lexical retrieval locally.
* Configurable embeddings.
* Local embedding option where practical.
* Vector similarity.
* Entity and symbol matching.
* File-path and artifact overlap.
* Memory-type signals.
* Episode and task-state signals.
* Temporal relevance.
* Importance.
* Confidence.
* Evidence quality.
* Branch or execution scope where represented.
* Duplicate penalties.
* Staleness and contradiction penalties.

Implement:

```text
task understanding
→ hard namespace and permission filtering
→ parallel candidate generation
→ eligibility filtering
→ scoring
→ optional reranking
→ diversification
→ context construction
→ abstention
```

Requirements:

* Scope filtering happens inside storage queries.
* Retrieval produces component scores and exclusion reasons.
* Weights and thresholds are configurable.
* No trained-weight claims without training data.
* Graph relationships may be implemented as relational tables and traversal signals.
* Do not add a dedicated graph database unless controlled benchmarks prove that it materially improves results enough to justify operational complexity.
* Wrong, disputed, stale, or unsupported memory must not be presented as certain current truth.

### 6. Context construction

Return structured, token-bounded context containing only useful information:

* Task interpretation.
* Current state.
* Relevant decisions.
* Previous attempts.
* Known failures.
* Constraints.
* Unfinished work.
* Open questions.
* Related artifacts.
* Evidence citations.
* Uncertainty and conflict warnings.
* Selection diagnostics.
* Token count.

Implement:

* Automatic abstention.
* Configurable token budgets.
* Deduplication.
* Compression without unsupported claims.
* Historical retrieval when explicitly requested.
* Citation resolution.
* Inspectable provenance.
* Feedback collection.
* No-context behavior for irrelevant or trivial requests.

### 7. Storage and processing

Local V1:

* SQLite.
* WAL mode.
* FTS5.
* Transactional migrations.
* Foreign keys.
* Explicit application lifecycle.
* Integrity checks.
* Deterministic safe repair where possible.
* Backup, export, import, and deletion.
* No writes during module import.

Background processing:

* Durable jobs.
* At-least-once execution.
* Idempotent effects.
* Leases.
* Heartbeats where useful.
* Retry classification.
* Exponential backoff with limits.
* Dead-letter state.
* Crash recovery.
* Cancellation.
* Processing metrics.
* No partial authoritative state after failure.

Hosted-ready storage:

* Implement a clean PostgreSQL storage path only after the local engine and storage contracts are stable.
* Use PostgreSQL FTS and pgvector where justified.
* Preserve identical behavioral contracts between SQLite and PostgreSQL.
* Provide transactional migrations.
* Add tenant-safe database queries.
* Do not introduce Redis, Kafka, or multiple authoritative databases unless load evidence demonstrates necessity.

### 8. Public service

Provide a stable versioned API for:

* Event ingestion.
* Batch ingestion.
* Processing.
* Memory search.
* Context retrieval.
* Memory inspection.
* Memory history.
* Evidence inspection.
* Memory invalidation.
* Feedback.
* Health.
* Readiness.
* Integrity checks.
* Export.
* Deletion.

Requirements:

* Strict request and response schemas.
* Consistent errors.
* Pagination.
* Request IDs.
* Timeouts.
* Cancellation where possible.
* Idempotency.
* Generic cross-namespace not-found behavior.
* OpenAPI documentation.
* Explicit API versioning.
* Application factory.
* Clean startup and shutdown.
* No accidental local database creation.
* Authentication boundary suitable for hosted deployment.
* Namespace authorization policy at the service boundary without weakening engine-level isolation.

### 9. Observability and debugging

Record and expose:

* Why an event was accepted or rejected.
* Why a memory was proposed.
* Exact supporting evidence.
* Why a candidate was accepted or rejected.
* Why reconciliation selected an action.
* Why a memory was retrieved.
* Component retrieval scores.
* Why candidates were excluded.
* Why context abstained.
* Model, provider, prompt, and schema versions.
* Latency.
* Token usage.
* Estimated cost.
* Job attempts and failures.
* Storage and index health.

Logs must be structured, redacted, bounded, and safe.

### 10. Privacy, isolation, and deletion

Prove:

* Namespace isolation across all repositories, indexes, jobs, APIs, caches, and diagnostics.
* Cross-namespace direct-ID access cannot reveal existence.
* Evidence cannot support memory in another namespace.
* FTS and vector retrieval cannot leak across namespaces.
* Redaction covers events, memories, versions, spans, jobs, errors, logs, indexes, context, provider requests, and audit records.
* Secret values do not remain in SQLite, WAL, journal files, PostgreSQL test storage, logs, or exported data.
* Prompt injection inside stored evidence cannot modify system behavior.
* Namespace deletion removes or cryptographically makes inaccessible all owned data according to documented behavior.
* Export/import preserves evidence, versions, history, and identity safely.
* Direct tampering is detected by integrity tooling where prevention is impossible.

### 11. Evaluation framework

Use the same production ingestion, extraction, reconciliation, retrieval, and context paths for evaluation. Do not build benchmark-only implementations.

Build deterministic labelled datasets for:

* Memory extraction.
* Evidence attribution.
* Reconciliation.
* Temporal reasoning.
* Contradiction handling.
* Retrieval relevance.
* Stale-memory rejection.
* Abstention.
* Project isolation.
* Prompt injection.
* Secret handling.

Measure:

* Extraction precision and recall.
* Evidence attribution accuracy.
* Unsupported-memory rejection.
* Reconciliation accuracy.
* Temporal-state accuracy.
* Recall@k.
* Precision@k.
* NDCG.
* Stale-memory rejection.
* Abstention accuracy.
* Cross-namespace leakage.
* Context token efficiency.
* Ingestion latency.
* Processing latency.
* Retrieval latency.
* Storage growth.
* Token and monetary cost.

### 12. LongMemEval-s

Implement a reproducible LongMemEval-s runner.

Freeze and report:

* Dataset revision.
* TermyteDB commit.
* Schema version.
* Prompts and hashes.
* Extraction model.
* Embedding model.
* Reranker.
* Answer model.
* Retrieval weights.
* Top-k settings.
* Token budgets.
* Hardware.
* Cache state.
* Latency.
* Tokens.
* Cost.

Compare fairly against feasible baselines under identical conditions:

* No memory.
* Raw history.
* Summary memory.
* Vector-only retrieval.
* TermyteDB.
* Mem0, Graphiti, or other researched systems when reproducible locally.

Do not quote incomparable published scores as direct evidence.

### 13. Coding-agent continuation benchmark

This is the primary product evaluation.

Create real, reproducible cases containing:

* Repository snapshot.
* Initial task.
* Agent A trajectory.
* Decisions.
* Discoveries.
* Failed approaches.
* Resulting repository state.
* Continuation task for Agent B.
* Expected behavioral outcome.
* Verification tests.

Compare:

* No memory.
* Raw transcript.
* Previous-session summary.
* Vector-only memory.
* TermyteDB.
* Oracle context.

Measure:

* Task completion.
* Tests passed.
* Time.
* Tool calls.
* Tokens.
* Repeated mistakes.
* Incorrect assumptions.
* Unnecessary file reads.
* Context precision.
* Context utilization.
* Cost.

Start with a small high-quality dataset and expand it. Run repeated trials where nondeterminism affects conclusions. Clearly separate actual results from untested claims.

### 14. Performance and reliability

Create local performance and load benchmarks for:

* Single event ingestion.
* Batch ingestion.
* Job throughput.
* Extraction overhead.
* FTS retrieval.
* Vector retrieval.
* Hybrid retrieval.
* Context assembly.
* Concurrent namespace usage.
* Restart recovery.
* Storage growth.

Profile real bottlenecks before optimizing.

Establish realistic V1 targets from measured baselines. Optimize only where measurements justify it. Preserve correctness, evidence integrity, and isolation over raw speed.

Test:

* Large histories.
* Large tool outputs.
* Concurrent ingestion.
* Concurrent processing.
* Worker crashes.
* Database restarts.
* Corrupted records.
* Missing indexes.
* Expired leases.
* Provider outages.
* Timeouts.
* Invalid model output.
* Partial migrations.
* Interrupted import/export.
* Disk errors where safely simulatable.

### 15. Developer experience

Provide:

* Reliable installation.
* Clear configuration.
* Example local setup.
* Example embedded Python use.
* Example HTTP use.
* Generated OpenAPI schema.
* Database initialization command.
* Worker command.
* Health and integrity commands.
* Benchmark commands.
* Export/import commands.
* Migration commands.
* Development setup.
* Test commands.
* Troubleshooting guide.
* Security and privacy documentation.
* Architecture documentation.
* Provider configuration documentation.
* Complete examples using synthetic data.

Do not require an external account for local deterministic operation.

## Autonomous milestone policy

Create or update a living execution plan with dependency-ordered milestones and explicit validation commands.

For each milestone:

1. Inspect the current implementation and tests.
2. Define the smallest coherent vertical slice.
3. Record acceptance criteria.
4. Implement it.
5. Format, lint, type-check, and test.
6. Add adversarial and regression coverage.
7. Run the end-to-end path.
8. Inspect actual storage and outputs.
9. Repair failures before continuing.
10. Update documentation and ADRs.
11. Commit the milestone separately.
12. Audit whether the overall Goal is complete.
13. If incomplete, automatically begin the next highest-value unblocked milestone.

Do not pause merely because one milestone finished.

Do not rewrite working foundations without evidence.

Do not hide failing tests, lower assertions, delete difficult cases, weaken isolation, or redefine acceptance criteria to manufacture success.

Do not commit secrets, generated databases, credentials, caches, benchmark datasets with incompatible licenses, or unnecessary model outputs.

Use deterministic fakes for automated tests. Make live-provider and paid-model evaluations explicit and optional. Use normally configured credentials only. Never invent credentials or expose their values.

Keep the Git working tree understandable. Preserve user changes. Make focused milestone commits. Never use destructive Git operations.

## Decision policy

When the repository documents already determine a choice, follow them.

When implementation reveals a conflict:

1. Gather evidence from source, tests, and measurements.
2. Compare viable alternatives.
3. Choose the smallest option that preserves correctness and buildability.
4. Record the decision in an ADR.
5. Continue without asking unless the decision requires credentials, external authority, destructive action, substantial cost, or materially changes the product definition.

When an experiment can answer uncertainty, run the experiment instead of extending speculative planning.

Use the previously researched Supermemory, Cognee, TencentDB Agent Memory, Mem0, and Graphiti repositories as read-only implementation references when useful. Do not copy code without verifying its license, preserving required attribution, and recording reuse. Prefer independent implementation around TermyteDB’s own contracts.

## V1 non-goals

Do not build:

* Termyte CLI coding-agent hooks.
* Termyte SDK framework adapters.
* Slack, GitHub, Linear, Jira, Gmail, Notion, or document connectors beyond generic canonical ingestion.
* A web dashboard.
* Enterprise SAML, SCIM, regional hosting, compliance certification, or VPC deployment.
* Agent orchestration.
* Task management.
* A generic code-search product.
* A dedicated graph database without benchmark proof.
* Large provider matrices.
* Infrastructure unsupported by measured need.

These belong after TermyteDB V1 proves memory quality.

## Completion contract

The Goal is complete only when all of the following are true:

* The complete evidence-to-memory-to-context system works end to end.
* Imports create no files or external effects.
* Local deterministic operation requires no account or network.
* Model output cannot directly write authoritative memory.
* Every accepted memory has valid persisted evidence.
* Unsupported memories are rejected.
* Memories are versioned and auditable.
* Corrections, supersession, disputes, and historical truth behave correctly.
* Namespace isolation is enforced below the prompt layer and passes adversarial tests.
* Ingestion and processing are idempotent under retries and concurrency.
* Crashes and restarts do not corrupt authoritative state.
* FTS and vector retrieval work through one explainable pipeline.
* Context is token-bounded, cited, uncertainty-aware, and capable of abstaining.
* SQLite local operation is reliable.
* PostgreSQL behavior is implemented and contract-tested if the build plan classifies it as V1 rather than hosted post-V1.
* The versioned HTTP API is documented and tested.
* Integrity, export, import, invalidation, and deletion work.
* Extraction, reconciliation, retrieval, temporal, isolation, and redaction evaluations are reproducible.
* LongMemEval-s runs reproducibly or is documented as blocked by a specific external dependency after all repository-local work is complete.
* The coding-agent continuation benchmark runs with real verification tests.
* TermyteDB demonstrates measurable improvement over at least the no-memory and previous-summary baselines on the available coding benchmark; if it does not, continue diagnosing and improving the system rather than hiding the result.
* Performance and storage behavior are measured.
* Ruff formatting and linting pass.
* mypy passes.
* The full automated test suite passes.
* End-to-end, concurrency, restart, crash-recovery, isolation, redaction, migration, integrity, and benchmark tests pass.
* Documentation matches actual behavior.
* Installation works from a clean environment.
* The final working tree is clean.
* Milestones are committed separately.
* A final release-readiness report clearly separates verified capabilities, measured results, remaining limitations, and post-V1 work.

If the Goal cannot be fully completed because of a real blocker, exhaust safe repository-local alternatives first. Stop only when progress requires user credentials, paid external usage, unavailable data, external authorization, destructive action, or a product decision that cannot be resolved from existing evidence. Report the exact blocker, evidence, completed work, current test state, and smallest action required from the user.

Do not declare success because TermyteDB contains many features. Declare success only when it is demonstrably useful, reliable, isolated, evidence-backed, reproducible, and buildable.
