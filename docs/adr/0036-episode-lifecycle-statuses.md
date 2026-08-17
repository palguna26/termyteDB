# ADR 0036: Episode lifecycle statuses

## Decision

Episodes support the constrained statuses `active`, `completed`, `failed`,
`abandoned`, and `interrupted`.

The HTTP request validates the status before it reaches the repository. The
update is scoped by both episode ID and namespace ID, so an episode cannot be
changed through another namespace. Optional summaries are redacted before
storage.

## Reason

An episode needs an explicit end state so clients can distinguish work that
finished, failed, was abandoned, or was interrupted. Keeping the status set
small makes the lifecycle stable for API consumers.
