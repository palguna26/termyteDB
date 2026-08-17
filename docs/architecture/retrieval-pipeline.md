# Retrieval pipeline

```text
understand task -> enforce scope/policy -> parallel candidates
 -> remove invalid/stale/unsupported -> explainable rerank
 -> diversify -> bounded context -> abstain when support is weak
```

Candidate sources in V1 are SQLite FTS5 and optional embeddings. Metadata and validity predicates apply in SQL before candidates leave the scope. Graph expansion is not required. Initial score is explainable: normalized lexical score 0.45, vector score 0.35, recency/validity 0.10, evidence confidence 0.10. These are starting constants, not trained weights; benchmark data may change them.

Tie-breaking is deterministic. Candidates are grouped by memory subject and only the best current version is selected unless the request asks for history/conflicts. Diversification limits near-duplicate statements and caps one episode's contribution. Context construction includes statement, status, time, confidence, evidence IDs, and compact source excerpts until the token budget is reached. If no candidate passes support, scope, and score thresholds, return `abstained=true` with diagnostics.

