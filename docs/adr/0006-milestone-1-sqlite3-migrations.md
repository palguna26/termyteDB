# ADR 0006: Milestone 1 uses sqlite3 with transactional migrations

Status: accepted for Milestone 1.

The broader architecture names SQLAlchemy/Alembic as the hosted-ready target. The first vertical slice uses Python's standard `sqlite3` transaction API and an explicit `schema_migrations` table instead. This keeps the SQLite-only milestone small, makes the authoritative SQL visible, and avoids adding an ORM before the domain and query predicates are proven. The migration boundary is isolated in `termytedb/db.py` so SQLAlchemy/Alembic can replace it in a later hosted milestone without changing the engine API.

Rejected for this milestone: adding SQLAlchemy/Alembic only as unused scaffolding. This is a bounded implementation deviation, not a change to the storage decision.

