# Conversational Memory LongMemEval Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

## Objective

Implement the phase-1 conversational memory system described in `docs/superpowers/specs/2026-08-27-conversational-memory-longmembench-design.md`.

Primary outcome:
- beat the current Supermemory LongMemEval-S baseline in every category
- keep one unified conversational pipeline
- keep graph support minimal and secondary
- keep product fallback paths available
- keep the benchmark path OpenRouter-first

## Non-negotiable benchmark rule

Benchmark loops are **never automatic**.

- The default benchmark loop size is a **subset of 5 samples out of 500**
- Those 5 samples are only used for smoke testing and tuning checks
- Before any benchmark loop is run, ask the user first
- Do not start a benchmark loop without explicit user approval
- Full 500-sample runs remain available, but only after a separate user approval

This rule applies to:
- LongMemEval-S tuning loops
- retrieval regressions
- context packing regressions
- extraction regression checks

## Phase breakdown

### Phase 1A: lock the benchmark and product contracts

Goal:
- make the benchmark and product paths explicit in code and docs
- separate the OpenRouter benchmark path from local fallback recovery
- make the subset-of-5 benchmark smoke loop the default tuning unit

Work:
1. Update benchmark config loading so the benchmark mode can be run in a 5-sample smoke configuration.
2. Make the benchmark runner require an explicit user-triggered flag or command path before it runs.
3. Make the output schema include:
   - sample count
   - category
   - token usage
   - latency
   - accepted/rejected counts
   - final score fields
4. Ensure local fallback never becomes the benchmark authority.
5. Keep the product runtime able to fall back locally for dev and recovery.

Exit criteria:
- benchmark path and product path are explicit
- 5-sample smoke runs are supported
- benchmark execution is always manual

### Phase 1B: make ingestion conversational-only

Goal:
- remove tool-agent behavior from the memory layer
- keep extraction focused on conversation text only

Work:
1. Keep payload flattening centered on messages, user text, assistant text, and conversation metadata.
2. Ignore execution-only fields in the extraction path.
3. Preserve stable evidence offsets for conversational content.
4. Keep raw event storage unchanged.
5. Keep redaction before any model call.

Exit criteria:
- extractor output contains only conversational material
- no tool-trace text enters the memory pipeline
- existing conversational tests pass

### Phase 1C: harden extraction and validation

Goal:
- improve memory quality before retrieval tuning

Work:
1. Keep OpenRouter extraction as the benchmark path.
2. Keep the local rule extractor only as fallback.
3. Keep validation strict:
   - evidence span checks
   - semantic support checks
   - secret redaction checks
   - unsupported claim rejection
4. Keep candidates atomic and self-contained.
5. Keep subject normalization and statement normalization stable.

Exit criteria:
- accepted candidates are grounded
- unsupported claims are rejected
- extraction errors are visible in run diagnostics

### Phase 1D: implement temporal memory behavior

Goal:
- make versioning and temporal correctness first-class

Work:
1. Keep versioned memories as canonical state.
2. Preserve valid-from and valid-until.
3. Keep supersession explicit.
4. Demote invalidated and superseded memories in ranking.
5. Prefer the newest valid fact on conflict-heavy queries.
6. Keep historical mode available for ablations and inspection.

Exit criteria:
- knowledge-update cases can represent a replacement cleanly
- temporal reasoning queries can see correct ordering
- restore/invalidate paths preserve history

### Phase 1E: dense retrieval and sqlite-vec

Goal:
- keep dense retrieval fast, indexed, and recoverable

Work:
1. Keep `memory_embeddings` as authoritative storage.
2. Use `sqlite-vec` as the fast retrieval index.
3. Keep NumPy fallback only as a recovery path.
4. Rebuild the vec index from canonical embeddings on demand.
5. Keep provider and dimension tracking strict.
6. Keep namespace scoping strict.

Exit criteria:
- dense search runs through sqlite-vec when available
- fallback search remains correct
- rebuilds produce consistent results

### Phase 1F: hybrid retrieval and ranking

Goal:
- maximize Recall@15 with aggregation without overfitting to one category

Work:
1. Keep lexical FTS in the retrieval stack.
2. Keep dense retrieval in the retrieval stack.
3. Fuse lexical and dense signals into one ranking.
4. Add temporal validity filters before final ranking.
5. Keep exact-match and evidence-support boosts.
6. Keep confidence, importance, and recency signals modest.
7. Keep graph proximity only as a secondary ranking signal.

Exit criteria:
- retrieval remains hybrid
- ranking is stable across categories
- category regressions are visible in smoke runs

### Phase 1G: minimal graph index

Goal:
- preserve relation signals without turning the system into a graph-first product

Work:
1. Store only explicit edges in phase 1.
2. Persist edges in SQLite.
3. Keep a rebuild path from memory versions.
4. Keep edge types limited to:
   - memory-to-memory
   - entity-to-entity
   - session adjacency
5. Keep edges unweighted in phase 1.
6. Use graph only as a ranking signal.

Exit criteria:
- graph edges can be rebuilt
- graph edges influence ranking without owning retrieval
- no graph traversal dependency is needed for correctness

### Phase 1H: session summaries

Goal:
- help multi-session recall and context packing without replacing atomic memories

Work:
1. Create session summaries with an LLM.
2. Generate summaries only at session end.
3. Store summaries as separate memory artifacts.
4. Use summaries to support retrieval and packing.
5. Keep summaries from overriding atomic facts.

Exit criteria:
- summaries improve recall or packing quality
- summaries do not create duplicate authority problems

### Phase 1I: context packing

Goal:
- keep the context budget small while preserving the best recall

Work:
1. Remove duplicate results.
2. Keep provenance visible.
3. Keep token budgeting explicit.
4. Prefer the highest-value results first.
5. Keep packed context compact enough for LongMemEval-S aggregation.

Exit criteria:
- packed context stays bounded
- token usage is tracked per sample
- packing changes are measurable in smoke runs

### Phase 1J: benchmark tuning loop

Goal:
- tune against the benchmark without turning the benchmark into an automatic background task

Work:
1. Use the 5-sample smoke subset for rapid tuning checks.
2. Ask the user before every benchmark loop.
3. Compare per-category results against the current baseline.
4. Record regressions and improvements after each change.
5. Only escalate to full 500-sample runs when the user approves.

Exit criteria:
- tuning loops are manual
- 5-sample checks are fast and repeatable
- benchmark runs are not started implicitly

## Implementation order

1. benchmark contract and CLI gating
2. conversational-only extraction boundary
3. validation and temporal correctness
4. dense retrieval and sqlite-vec
5. hybrid ranking and packing
6. minimal graph index
7. session summaries
8. smoke benchmark loop
9. category tuning
10. full benchmark only after approval

## Verification plan

Run these checks during implementation:
- focused unit tests for extraction and validation
- dense retrieval tests with sqlite-vec available and unavailable
- temporal supersession and restore tests
- graph rebuild tests
- session summary write tests
- 5-sample LongMemEval smoke runs only after user approval

## Deliverables

1. Updated extraction and retrieval code
2. Minimal graph index support
3. Session summary support
4. Benchmark runner support for the 5-sample smoke loop
5. Manual benchmark gating
6. Updated docs describing the benchmark and product contracts

