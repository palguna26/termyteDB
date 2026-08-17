# ADR 0033: Optional HTTP namespace rate boundary

## Status

Accepted

## Decision

The service factory accepts `rate_limit_per_minute`. When configured, a thread-safe sliding window is maintained independently per namespace and rejects excess event, batch-event, processing, search, and context requests with HTTP 429 and `Retry-After`. The default is disabled so embedded deterministic use has no hidden process-global limit.

## Consequences

Local deployments can enforce a simple bounded request rate without adding a queue or external state store. Hosted deployments needing distributed quotas must replace this callback-level limiter with an external policy layer after load evidence.
