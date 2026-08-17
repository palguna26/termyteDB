# ADR 0010: Namespace-scoped event identity

Status: accepted for Milestone 1.1.

Event identity is UUID5 over `(namespace_id, idempotency_key)`. The database uniqueness constraint is `(namespace_id, idempotency_hash)`. Canonical content hashes include event type, stream, occurred time, and redacted payload with sorted JSON keys. Reusing a key with different content returns a conflict; retrying identical content returns the original event and job.

Concurrent writes serialize through the database lock and the uniqueness constraint, producing one event and one job.

