# ADR 0046: Paginated memory inspection

## Decision

Expose `GET /v1/memories` with bounded namespace-scoped `limit` and `offset`.
Each item uses the same current-version, status, confidence, and citation
projection as direct memory inspection.

## Reason

Operators need collection inspection for audits and clients need pagination;
direct-ID lookup alone cannot support either use case safely at scale.
