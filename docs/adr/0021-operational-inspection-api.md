# ADR 0021: Operational inspection API

## Status

Accepted

## Decision

The versioned local API exposes namespace-scoped event/evidence inspection and processing-job inspection, plus a database integrity report. Direct IDs require the namespace and return the same not-found response across namespaces. Integrity reports use the existing SQLite foreign-key, integrity, evidence, FTS, and schema checks.

## Consequences

Operators can verify provenance, job state, and storage health without opening SQLite directly. The integrity endpoint is diagnostic and does not silently repair data; deterministic repair remains an explicit command.
