# ADR 0054: Concurrent worker processing

## Decision

Concurrent workers may claim jobs for the same namespace. SQLite transactions,
leases, and namespace predicates must ensure each job is completed once and
each event produces one authoritative memory effect.

## Reason

At-least-once processing is useful only if overlapping workers do not duplicate
authoritative versions. The worker behavior is now covered by a 20-job,
two-worker regression test.
