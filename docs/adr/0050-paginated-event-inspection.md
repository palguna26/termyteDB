# ADR 0050: Paginated event inspection

## Decision

Expose `GET /v1/events` with bounded namespace-scoped pagination. Each event
uses the existing redacted direct-inspection projection, including artifacts
and evidence references.

## Reason

Evidence review requires collection access, while direct-ID inspection remains
useful for precise lookup. Reusing the projection avoids a second redaction or
serialization path.
