# ADR 0015: Local lifecycle and batch API

## Status

Accepted

## Decision

The local V1 service exposes batch event ingestion, memory history, explicit invalidation, namespace export, namespace deletion, health, and readiness endpoints. These operations use the same namespace predicates as the embedded repository and do not create a second source of truth.

Batch ingestion reuses the idempotent single-event path so retries preserve the established conflict behavior. Namespace deletion removes authoritative rows and rebuildable FTS rows in one transaction. Historical memory versions remain inspectable until their namespace is deleted or the owning policy removes them.

## Consequences

The API now supports the minimum operational lifecycle needed for local testing and deployment checks. Import, hosted PostgreSQL, authentication, and vector indexes remain separate milestones because they need their own contracts and tests.
