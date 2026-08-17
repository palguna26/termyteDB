# ADR 0055: Thin SDK clients

## Decision

Ship small Python and TypeScript HTTP clients around the versioned service
contract. They use standard runtime HTTP facilities, send request IDs, enforce
timeouts, retry only transient responses, and expose structured errors.

## Reason

SDK readiness is the next dependency-ordered build milestone. Keeping the
clients thin avoids duplicating engine behavior and keeps local installation
dependency-free while giving embedded and network users the same API contract.
