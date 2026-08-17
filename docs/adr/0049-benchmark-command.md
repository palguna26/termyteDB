# ADR 0049: Benchmark command

## Decision

Expose the existing local production benchmark as
`python -m termytedb.operations benchmark --events N`. It prints JSON metrics
for ingest, processing, retrieval, context, restart, and concurrent namespace
usage.

## Reason

Performance evidence should be reproducible from the same production paths,
without requiring users to write a Python harness or run an external service.
