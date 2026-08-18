# ADR 0062: Persisted retry backoff

## Decision

Failed jobs remain retryable but receive a persisted `next_attempt_at`. The
delay grows exponentially from one second and is capped at five minutes;
dead-letter jobs have no next attempt.

## Reason

Immediate retries can hot-loop during provider outages and waste worker
capacity. Persisting the schedule makes backoff survive restarts and keeps
claiming deterministic across workers.
