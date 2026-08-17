# ADR 0008: At-least-once processing with idempotent effects

Status: accepted for Milestone 1.1.

Jobs are durably recorded before processing. A worker claims a job with a lease, may execute it more than once, retries failures up to the configured limit, reclaims expired leases, and marks exhausted jobs `dead`. Memory-version creation, evidence references, and FTS updates commit atomically. Repeating a committed candidate does not create a duplicate version or evidence reference. A crash can therefore repeat execution, but not committed effects.

TermyteDB makes no exactly-once execution claim. It provides at-least-once execution with idempotent processing effects and durable dead-letter state.

