# ADR 0056: Active job heartbeats

## Decision

Workers extend the lease of a claimed job through a namespace-scoped heartbeat
before extraction. The update is accepted only while the job remains in the
`processing` state.

## Reason

Provider calls can last longer than the initial lease. Extending an active
lease reduces duplicate claims while preserving cancellation and completion
guards; it does not revive completed, cancelled, or cross-namespace jobs.
