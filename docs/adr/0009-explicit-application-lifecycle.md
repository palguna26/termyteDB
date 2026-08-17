# ADR 0009: Explicit application lifecycle

Status: accepted for Milestone 1.1.

Importing `termytedb` modules performs no database construction or file I/O. `TermyteDB` and `create_app` require an explicit database path or injected `Database`. FastAPI lifespan closes the engine and checkpoints SQLite on shutdown. The command entry point requires `--database`.

Rejected: a module-level default app or implicit `termytedb.sqlite` path.

