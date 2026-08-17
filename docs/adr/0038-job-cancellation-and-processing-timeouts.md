# ADR 0038: Job cancellation and processing timeouts

## Decision

Jobs can be cancelled through a namespace-scoped API path. Processing also
accepts a bounded timeout and stops claiming work after the deadline; a job
already being processed is allowed to finish its current synchronous unit.

## Reason

Cancellation must be durable so a later worker does not claim the job again.
The timeout is a processing boundary, not a forced thread kill, which keeps
authoritative transactions consistent and avoids partial memory writes.
