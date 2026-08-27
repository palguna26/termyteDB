# Conversational Memory LongMemEval Design

## Goal

Build TermyteDB as a conversational memory infrastructure layer that can beat the current Supermemory LongMemEval-S baseline in every category.

The system must:
- stay conversational-first
- use one unified retrieval pipeline
- use a minimal graph index as a secondary signal only
- keep local fallback for development and recovery
- force the benchmark path through OpenRouter
- preserve versioned memory history and temporal validity

This spec locks the phase-1 product shape. It does not add tool-agent support, graph-first traversal, or distributed hosting.

## Benchmark contract

The benchmark path is the product path under test, not a separate toy pipeline. It must measure the same memory flow that production uses, with one exception: local fallback does not count toward benchmark scoring.

Benchmark rules:
- LongMemEval-S only
- Recall@15 with aggregation as the primary retrieval metric
- category scores reported for:
  - single-session user
  - single-session assistant
  - single-session preference
  - knowledge update
  - temporal reasoning
  - multi-session
- token usage tracked per sample
- latency tracked per stage
- OpenRouter extraction required
- OpenRouter embeddings required
- sqlite-vec used when available
- local fallback allowed for failure recovery, but not as the benchmark target

The benchmark result file must include configuration, timing, candidate counts, accepted/rejected counts, packed token usage, and per-category totals.

## Product contract

The product contract is a conversational memory layer that other apps can build on top of.

It must support:
- ingesting conversational events
- extracting atomic memories from conversation text
- validating candidate memories against evidence
- versioning memories instead of overwriting them
- retrieving memories through hybrid lexical + dense search
- packing bounded context for downstream use
- exposing provenance and citations

The product contract keeps a local fallback path for:
- developer use
- offline use
- provider failure recovery

That fallback is a runtime safety path, not the benchmark target.

## Architecture

The system is split into five layers.

### 1. Ingestion layer

The ingestion layer accepts conversational payloads and stores them as immutable source events.

Responsibilities:
- validate namespace and idempotency
- redact before model calls
- store the raw event
- create a processing job
- assign the event to a session boundary
- preserve artifacts and metadata

Non-goals:
- tool execution tracing
- agent orchestration
- arbitrary event adapters in the core memory flow

### 2. Extraction layer

The extraction layer turns conversation text into candidate memories.

Responsibilities:
- flatten conversation payloads into stable text
- ignore tool-specific execution fields
- call OpenRouter in the benchmark path
- call the local rule extractor only as fallback
- extract atomic candidate memories
- preserve exact evidence spans
- reject unsupported or ungrounded claims

Extraction output must stay small and self-contained:
- one claim per candidate
- one evidence span set per candidate
- one clear subject key per candidate

### 3. Memory store

The memory store is the source of truth.

Stored entities:
- `events`
- `memories`
- `memory_versions`
- `evidence_refs`
- `memory_embeddings`
- `session_summaries`
- `graph_edges`

Rules:
- never overwrite a memory version in place
- supersession creates a new version
- invalidation preserves history
- restore reactivates the newest valid version
- embeddings are stored authoritatively in SQLite BLOB form

### 4. Retrieval layer

Retrieval is hybrid and unified.

Signals:
- lexical FTS
- dense vector similarity
- temporal validity
- evidence support
- confidence
- importance
- recency
- graph proximity

Retrieval must not depend on graph traversal alone. The graph index is only one ranking signal.

### 5. Packing layer

Packing converts ranked memories into bounded context.

Responsibilities:
- remove duplicates
- preserve provenance
- enforce token budget
- keep category-sensitive context compact
- prefer the highest-value memories first

## Data flow

### Ingest path

1. The API receives a conversational event.
2. The event is redacted and validated.
3. The event is stored immutably.
4. A processing job is created.
5. The event is assigned to a session boundary.
6. The event becomes available for extraction.

### Process path

1. A worker claims jobs with a lease token.
2. The processor loads the event.
3. `payload_text()` converts the payload into conversation text.
4. The extraction provider returns candidate memories.
5. `validate_candidate()` checks evidence spans and semantic support.
6. Accepted candidates are embedded.
7. The repository writes memory versions, evidence refs, FTS rows, embeddings, and graph edges.
8. The job is marked complete or failed.

### Search path

1. The query is tokenized.
2. Lexical search runs over FTS.
3. Dense search runs over sqlite-vec.
4. If needed, dense fallback runs over stored embeddings.
5. Temporal and status filters remove invalid memories.
6. Candidate memories are fused and ranked.
7. A final bounded result list is returned with citations and component scores.

### Context path

1. Search returns ranked memories.
2. The packer removes duplicates and low-value items.
3. The packer trims to token budget.
4. The final context text and diagnostics are returned.

## Temporal handling

Temporal correctness is a first-class requirement.

Phase 1 rules:
- active memories win over superseded memories
- newest valid information wins when facts conflict
- valid-from and valid-until remain part of search filters
- history stays queryable through historical mode
- temporal tie-breaking must be explicit in the score

This is necessary for knowledge update and temporal reasoning categories.

## Dense retrieval

The dense retrieval stack must use:
- OpenRouter embeddings in the benchmark path
- `sqlite-vec` as the fast index
- `memory_embeddings` as the source of truth
- NumPy fallback only when the vec path is unavailable

The vec index is a derived acceleration structure, not the canonical store.

Requirements:
- one vector per memory version
- one embedding model per benchmark run
- explicit provider and dimension tracking
- automatic rebuild if dimensions change
- namespace-aware filtering

## Minimal graph index

Phase 1 includes a minimal graph index, but it stays secondary.

Graph rules:
- explicit links only in phase 1
- no inferred coreference in phase 1
- no graph-first traversal in phase 1
- graph edges act as ranking signals only
- graph data must be rebuildable from memory versions
- edges persist in SQLite and can be regenerated

Stored edge types:
- memory-to-memory
- entity-to-entity
- session adjacency

Entity linking:
- extraction-first for explicit links
- small high-confidence linker for repeated names

Session adjacency:
- same-session edges
- narrow rolling cross-session edges

Edges are unweighted in phase 1. Confidence and recency weights can be added later if they improve the benchmark.

## Session summaries

Session summaries are first-class memory artifacts.

Phase 1 rules:
- summaries are generated with an LLM
- summaries are created at session end only
- summaries are stored separately from atomic memories
- summaries are used as retrieval support, not as a replacement for atomic extraction

Purpose:
- improve multi-session recall
- reduce packing cost
- preserve the user-level story of a session

## Retrieval scoring

Final scoring should combine:
- lexical rank
- dense rank
- temporal validity
- exact match
- citation support
- memory confidence
- memory importance
- graph proximity

The retrieval stack should prefer:
- exact category hits
- newer valid facts for conflict-heavy questions
- memories with stronger evidence support
- memories that have already proven useful in retrieval

The score model must be simple enough to tune against benchmark traces.

## Error handling

### Extraction failures

If OpenRouter extraction fails:
- retry transient failures
- record run failure metadata
- fall back to local rule extraction for product recovery only
- do not count fallback as benchmark success

### Embedding failures

If OpenRouter embeddings fail:
- retry transient failures
- keep the stored memory version
- allow local embedding fallback for recovery
- mark the benchmark path as failed if the run depends on fallback

### Vector index failures

If sqlite-vec is unavailable or invalid:
- rebuild from canonical embeddings when possible
- fall back to NumPy scoring
- keep search available

### Graph index failures

If graph edges are missing or stale:
- rebuild from memory versions
- do not block retrieval
- treat graph as a secondary signal only

### Job failures

Jobs must keep lease fencing, retry limits, and dead-letter handling.
Failed jobs must not create duplicate memory versions.

## Testing strategy

### Unit tests

Cover:
- payload flattening for conversational-only input
- candidate validation and evidence support
- temporal supersession rules
- memory version restore and invalidate behavior
- dense index bootstrap and fallback
- graph edge rebuild
- session summary write path

### Integration tests

Cover:
- ingest -> process -> search -> context
- OpenRouter path with mocked provider responses
- fallback path when OpenRouter or sqlite-vec is unavailable
- namespace isolation
- import/export rebuild behavior

### Benchmark tests

Cover:
- LongMemEval-S end-to-end run
- per-category scores
- token usage
- latency
- category regressions after tuning changes

## Non-goals for phase 1

- tool-agent memory support
- graph-first traversal
- distributed storage
- multi-tenant hosted scaling
- broad plugin or adapter ecosystem
- write-time knowledge graph inference beyond explicit links

## Phase 1 success criteria

Phase 1 is done when:
- the benchmark path is reproducible end to end
- every LongMemEval-S category beats the current Supermemory baseline
- temporal updates behave correctly
- token use stays competitive
- the graph index helps without becoming the core architecture
- local fallback remains available for development and recovery

## Implementation order

1. lock benchmark contracts and output schema
2. harden conversational extraction
3. tune OpenRouter extraction and embeddings
4. strengthen temporal ranking and supersession
5. improve hybrid retrieval and packing
6. add minimal graph edges and session summaries
7. run benchmark loops and tune against category regressions

