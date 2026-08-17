# ADR 0034: Collection pagination

## Status

Accepted

## Decision

Namespace-scoped jobs, episodes, feedback, and context-audit collections accept bounded `limit` (1-100) and non-negative `offset` query parameters. Defaults are safe for local use, SQL applies the namespace predicate before pagination, and invalid bounds return schema validation errors.

## Consequences

Inspection endpoints cannot accidentally return unbounded collections. Cursor pagination and total counts remain future improvements if measured workloads require them.
