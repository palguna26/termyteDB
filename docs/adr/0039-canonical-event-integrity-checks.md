# ADR 0039: Canonical event integrity checks

## Decision

Integrity checks recompute each persisted event's canonical content hash from
its stored identity, redacted payload, and artifact descriptors. Mismatches are
reported and make the integrity report unhealthy; repair does not rewrite
immutable evidence.

## Reason

SQLite permissions cannot prevent every direct tampering scenario. Detecting a
changed payload or artifact descriptor gives operators evidence of corruption
without silently changing the source record.
