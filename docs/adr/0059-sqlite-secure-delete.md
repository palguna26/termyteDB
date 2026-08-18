# ADR 0059: SQLite secure deletion

## Decision

Enable SQLite `secure_delete` for local storage and verify namespace deletion
against the database, WAL, and journal files after checkpoint/close.

## Reason

Deleting rows is not enough if SQLite keeps old page contents readable. Secure
deletion strengthens the local privacy contract while preserving the existing
WAL and transactional lifecycle.
