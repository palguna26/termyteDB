# ADR 0011: SQLite integrity checks and deterministic redaction

Status: accepted for Milestone 1.1.

SQLite foreign-key checks, `integrity_check`, schema-version checks, orphan detection, and FTS consistency checks are exposed through `termytedb.integrity`. The only automatic repair is a deterministic FTS rebuild from active authoritative memory versions. Evidence and memory data are never guessed or repaired automatically.

Redaction runs before hashing, persistence, extraction, and model boundaries. Sensitive dictionary keys are redacted recursively. Job errors are redacted before persistence and logging. Shutdown checkpoints WAL; tests inspect the main database, WAL, journal, FTS rows, and captured logs.

