# Versioned public interface

The stable interface is framework-neutral and uses namespace-scoped IDs.

```http
POST /v1/events
POST /v1/events:batch
POST /v1/process
POST /v1/jobs/{job_id}/cancel
GET  /v1/jobs
POST /v1/search
POST /v1/context
GET  /v1/context/requests
GET  /v1/extraction/runs
GET  /v1/extraction/decisions
GET  /v1/events/{event_id}?namespace_id=...
GET  /v1/memories/{memory_id}?namespace_id=...
GET  /v1/memories/{memory_id}/history?namespace_id=...
POST /v1/memories/{memory_id}/invalidate
POST /v1/feedback
GET  /v1/feedback?namespace_id=...
GET  /v1/episodes?namespace_id=...
PATCH /v1/episodes/{episode_id}
GET  /v1/export?namespace_id=...
POST /v1/import?namespace_id=...
GET  /v1/integrity
GET  /health
GET  /ready
```

FastAPI generates the contract at `/openapi.json` and the interactive
documentation at `/docs`. The required versioned operation set is covered by
the service contract test.

Python:

```python
db = TermyteDB("memory.sqlite")
db.ingest({"namespace_id": "project:acme/api", "idempotency_key": "run-7-tool-3", "type": "tool.result", "payload": result})
db.process("project:acme/api")
answer = db.context("project:acme/api", "Why is auth using refresh tokens?", token_budget=1200)
```

TypeScript:

```ts
await client.ingest({ namespaceId, idempotencyKey, type: "decision", payload });
const context = await client.context({ namespaceId, query, tokenBudget: 1200 });
```

HTTP request bodies use the same JSON schemas. `process` is an operator/test convenience; hosted workers may process asynchronously. `search` returns candidates and scores; `context` returns selected evidence, provenance, exclusions, and abstention diagnostics.

Provider interfaces are limited to `Extractor.extract`, `Embedder.embed`, `Reranker.rank` (optional), `Storage`, and `Tokenizer.count`. No integration-specific types belong here.
