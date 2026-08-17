# ADR 0026: Versioned canonical event protocol

## Status

Accepted

## Decision

Canonical events use `event-v1` and persist namespace, stream, actor, agent, session, and source identities. These fields, occurrence time, type, and redacted payload are included in the deterministic content hash. Local persistence rejects redacted payloads larger than 1 MiB before creating an event or job.

## Consequences

Event identity and provenance remain stable across export/import and retries. Large content must be represented by a future content-addressed artifact reference instead of silently entering the event payload.
