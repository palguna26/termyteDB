# ADR 0058: Audit provider failures before structured output

## Decision

When a configured extraction provider fails before returning a response,
processing records a failed extraction run with provider/model identity,
redacted input metadata, and a machine-readable error class. No candidate or
authoritative memory is written.

## Reason

Job status alone does not explain provider failures in the extraction audit.
Recording the failed run preserves observability while keeping untrusted
provider output outside authoritative memory tables.
