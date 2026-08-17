# ADR 0041: SQLite online backup

## Decision

The local engine exposes an explicit backup operation backed by SQLite's
online backup API. It copies the live database, including committed evidence,
memory versions, jobs, and indexes, to a different destination without
requiring the source connection to close.

## Reason

Logical export is useful for portability, but operators also need a faithful
SQLite recovery copy. The destination must differ from the live database and
its parent is created when needed.
