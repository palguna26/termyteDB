# ADR 0025: Local performance benchmark

## Status

Accepted

## Decision

The local benchmark measures single-event ingest, batch ingest, processing, search, context assembly, and search after database restart using temporary SQLite storage. It also reports recovered job count and the storage/index configuration. The benchmark records observed values and does not turn a small local run into a universal capacity target.

## Consequences

Performance work can be based on repeatable measurements. Load, concurrency, disk failure, and large-history tests remain separate follow-up work when representative workloads are available.
