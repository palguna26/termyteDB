# Storage strategy

## Local MVP

SQLite is authoritative for all rows, WAL mode is enabled, and FTS5 indexes statement text and selected evidence text. Every memory version has a compact embedding behind a provider; retrieval requires a dense candidate and uses FTS5 as a lexical ranking signal. Brute-force search is acceptable for a bounded local alpha. A filesystem directory stores large artifacts with content hashes. One process owns the worker; restart scans leased jobs.

## Hosted production

PostgreSQL is authoritative; PostgreSQL FTS handles lexical search and pgvector is a rebuildable index. Object storage holds large artifacts. A managed queue is introduced only when measured throughput or worker isolation requires it. Tenant predicates and row-level security are defense in depth, not a replacement for application authorization.

Rejected: Neo4j/FalkorDB/Kuzu in V1 because Graphiti/Cognee demonstrate graph value but also a large operational surface; Redis/Kafka because Tencent's queue/recovery patterns are valuable but not necessary for a single-node alpha; multiple databases because authoritative duplication makes recovery and deletion harder. Add a graph index only after an ablation benchmark shows material Recall@k or coding continuation gain.
