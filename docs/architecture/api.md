# Initial public interface

The stable interface is framework-neutral and uses namespace-scoped IDs.

```http
POST /v1/namespaces/{namespace_id}/events
POST /v1/namespaces/{namespace_id}/process
POST /v1/namespaces/{namespace_id}/search
POST /v1/namespaces/{namespace_id}/context
GET  /v1/memories/{memory_id}
POST /v1/memories/{memory_id}/invalidate
POST /v1/feedback
```

Python:

```python
db.ingest(namespace_id="project:acme/api", event={"type":"tool.result", "payload": result}, idempotency_key="run-7-tool-3")
db.process(namespace_id="project:acme/api", mode="sync")
answer = db.context(namespace_id="project:acme/api", query="Why is auth using refresh tokens?", token_budget=1200)
```

TypeScript:

```ts
await client.ingest({ namespaceId, idempotencyKey, type: "decision", payload });
const context = await client.context({ namespaceId, query, tokenBudget: 1200 });
```

HTTP request bodies use the same JSON schemas. `process` is an operator/test convenience; hosted workers may process asynchronously. `search` returns candidates and scores; `context` returns selected evidence, provenance, exclusions, and abstention diagnostics.

Provider interfaces are limited to `Extractor.extract`, `Embedder.embed`, `Reranker.rank` (optional), `Storage`, and `Tokenizer.count`. No integration-specific types belong here.

