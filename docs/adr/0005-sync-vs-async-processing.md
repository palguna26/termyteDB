# ADR 0005: Synchronous durability, asynchronous intelligence

Status: accepted.

Validate, redact, hash, scope-check, and commit evidence synchronously. Extract, resolve, reconcile, embed, and index asynchronously with idempotent jobs. This preserves Tencent's fast append path and recovery semantics while avoiding model latency on ingest. Rejected: synchronous model calls in the write transaction.

