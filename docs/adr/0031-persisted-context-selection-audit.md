# ADR 0031: Persisted context selection audit

## Status

Accepted

## Decision

Every embedded context request gets a UUID and a namespace-scoped audit row containing the query, token budget, selected memory-version IDs, token count, abstention state, diagnostics, and timestamp. The API exposes these rows for authorized inspection. Audit rows are deleted with their namespace and included in canonical export/import.

## Consequences

Context selection is inspectable after the response and across restart. Audit data is diagnostic, not authoritative memory, and does not alter retrieval truth.
