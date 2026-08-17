# ADR 0045: Explicit historical retrieval

## Decision

Search and context accept a `historical` flag, defaulting to false. The
default returns only current, active, non-expired truth. An explicit historical
request may return superseded, disputed, invalidated, or expired versions and
keeps their status and evidence citations in the response.

## Reason

Historical reasoning is useful for audits and late evidence, but old knowledge
must never appear as current truth by accident.
