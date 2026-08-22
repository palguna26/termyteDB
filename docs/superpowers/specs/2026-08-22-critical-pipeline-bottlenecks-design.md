# Critical Pipeline Bottlenecks Design

## Scope

Fix five confirmed bottlenecks without changing TermyteDB's public event, memory, search, or context response contracts.

## Temporal retrieval

Normal retrieval must return a memory version only when it is active, belongs to an active memory, has not been superseded, has started, and has not expired. Both the embedding candidate query and the final candidate query will enforce `valid_from <= now` and the existing `valid_until > now` rule. Historical retrieval remains unrestricted by these active-time predicates.

## Reconciliation transitions

`update` and `supersede` will recognize correction terms plus common transition verbs: switched, moved, migrated, changed, no longer, now using, updated, prefer, renamed, deprecated, and stopped. Structured provider candidates with confidence at least 0.85 and explicit `update` or `supersede` intent will be trusted without a marker. Evidence, identity, and span validation remain mandatory.

## Binary vector storage

`memory_embeddings.vector_json` will be replaced by `memory_embeddings.vector BLOB`. A schema migration will convert existing JSON arrays to packed little-endian float32 bytes before removing the JSON column. New writes and index rebuilds will store float32 BLOBs only. Retrieval will filter vectors to the configured provider and dimensions, create NumPy views over the BLOB buffers, and score a batch with matrix multiplication. Historical embeddings remain stored.

## Token accounting

Context packing will use `tiktoken` when it is importable. The encoder will be cached. If it is unavailable or cannot initialize, counting will use `ceil(whitespace_words * 1.35)`. Headings and memory lines will use the same counter.

## Execution trace projection

The extraction projection will accept the event type as optional context. For `tool_execution`, `bash_command`, or payloads containing execution fields, it will preserve labeled tool name, command, stdout, stderr, exit code, error, and corrective or recovery text. Existing removal of wrapper-only environment blocks and system messages remains for ordinary conversational events.

## Compatibility and verification

Existing public method signatures remain compatible by making event type optional in extraction helpers. Regression tests will cover future validity, transition intents, binary migration and retrieval, token fallback, and execution traces. Focused tests will run first, followed by the full pytest suite.
