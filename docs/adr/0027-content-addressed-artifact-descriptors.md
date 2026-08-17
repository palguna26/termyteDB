# ADR 0027: Content-addressed artifact descriptors

## Status

Accepted

## Decision

Canonical events may carry up to 20 bounded artifact descriptors. Each descriptor contains a `sha256:` content address, media type, size, optional URI, and bounded metadata. Descriptors are stored in a namespace-scoped table linked to the immutable event; payload text remains subject to the 1 MiB event boundary. Artifact bytes are not copied into SQLite.

Artifacts are included in event hashing, export/import, event inspection, and namespace deletion. A future filesystem/object store can resolve the URI without changing the event contract.
