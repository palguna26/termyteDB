# ADR 0047: Namespace metrics

## Decision

Expose `GET /v1/metrics?namespace_id=...` with bounded aggregate counts for
events, memories, versions, jobs, extraction runs, decisions, job statuses,
and extraction latency. Every query is namespace-scoped.

## Reason

Operators need a compact health and throughput view without scraping logs or
reading raw tables. Metrics are derived from existing authoritative records and
do not introduce a second metrics database.
