# ADR 0003: SQLite first, PostgreSQL later

Status: accepted.

SQLite with FTS5 is the local authority. PostgreSQL/FTS/pgvector is the hosted target. Indexes are rebuildable. Rejected for V1: Neo4j, Redis, Kafka, and a multi-database authority set because their operational cost is not justified before benchmarks.

