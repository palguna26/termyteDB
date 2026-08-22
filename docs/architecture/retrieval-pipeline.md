# Retrieval pipeline

```text
understand task -> enforce scope/policy -> parallel candidates
 -> remove invalid/stale/unsupported -> explainable rerank
 -> diversify -> bounded context -> abstain when support is weak
```

Candidate sources in local V1 are SQLite FTS5 and optional rebuildable embeddings. Metadata and validity predicates apply before candidates leave the namespace. Graph expansion is not required.

Lexical and semantic results are ranked separately, then fused by rank. Exact term matches, evidence support, confidence, and importance provide small, visible boosts. A semantic-only result must clear a similarity floor; lexical matches do not depend on an embedding provider. If embedding generation is unavailable or fails, FTS results are still returned. Diagnostics show which retrieval modes contributed and the component scores for each result.

Tie-breaking is deterministic. Candidates are grouped by memory subject and only the best current version is selected unless the request sets `historical=true` on search or context. Historical results retain their superseded, disputed, invalidated, or expired status and citations. Diversification limits near-duplicate statements and caps one episode's contribution. Context is grouped by memory kind and includes statement, status, time, confidence, evidence IDs, and compact source excerpts until the token budget is reached. If no candidate passes support, scope, and score thresholds, return `abstained=true` with diagnostics.
