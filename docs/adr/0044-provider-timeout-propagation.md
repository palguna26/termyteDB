# ADR 0044: Provider timeout propagation

## Decision

Each provider extraction call receives the remaining processing deadline and a
cancellation callback. The generic HTTP provider uses the timeout for its
network request and checks cancellation before and after the request.

## Reason

A worker deadline is incomplete if a provider call can ignore it. Propagation
lets providers stop cooperatively while preserving transaction boundaries and
the existing retry classification.
