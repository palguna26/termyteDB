# ADR 0048: Local operations CLI

## Decision

Provide `python -m termytedb.operations` commands for database initialization,
namespace export/import, SQLite backup, and integrity checking/repair. Each
command requires explicit paths and namespace values where applicable.

## Reason

Local users should be able to operate the deterministic engine without writing
custom scripts or depending on hosted infrastructure. The commands call the
same production engine and storage operations used by the library and service.
