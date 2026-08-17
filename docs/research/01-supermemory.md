# Supermemory source review

## Confirmed implementation

The supplied TypeScript monorepo exposes a versioned HTTP contract in `packages/lib/api.ts` (`apiSchema`). It validates `POST /documents`, batch documents, document deletion, `POST /search`, container tags, processing status, and inferred-memory review. `packages/validation/api.ts` defines `MemoryAddSchema`, `SearchRequestSchema`, memory entries, embedding fields, version numbers, parent/child relations, metadata filters, and an optional `rerank` flag. `packages/ai-sdk/src/tools.ts` exposes add/search tools, so integrations are outside the core HTTP contract.

The visible source checkout is primarily frontend/API schema and client code; the persistence and server implementation are not present under the inspected packages. Therefore PostgreSQL, chunking, extraction prompts, and ranking internals cannot be confirmed from this checkout. `packages/lib/similarity.ts` confirms cosine similarity for available embeddings, but not the server's full retrieval algorithm.

The API uses `containerTag`/project-style grouping rather than a full organization-project-user-agent authorization model. The inferred-memory review endpoints show human review for some inferred memories. Delete endpoints exist, but an append-only evidence model and contradiction semantics are not established by the visible source.

## Engineering assessment

Strong decisions: typed API schemas; explicit memory/document distinction; embedding model fields; version and relation fields; batch ingestion; processing status; review queue; integration adapters kept in packages.

Weakness or unverified area: the supplied source does not prove immutable raw evidence, evidence-span attribution, temporal validity, storage-level authorization, or server-side failure recovery. These must not be copied into TermyteDB as assumptions.

## Reuse and rejection

Reuse the contract discipline, explicit processing state, review state, and relation-aware response shape. Reject the tag-only namespace as the authoritative security boundary; TermyteDB needs scope IDs in every table and query. Do not copy undocumented hosted internals. License is MIT (`LICENSE`); preserve notice and attribution if code is ever reused. No code is copied by this review.

