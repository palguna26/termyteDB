# ADR 0042: Injectable embedding provider

## Decision

The engine accepts an embedding provider with a name, dimension count, and
`embed` method. The deterministic local hash implementation remains the
default. Provider identity and dimensions are persisted with each vector.

## Reason

Retrieval needs a configurable embedding boundary without making network or
vendor dependencies part of local operation. Persisting provider metadata
makes index provenance inspectable and prevents an unlabelled model swap.
