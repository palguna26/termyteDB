# Cognee source review

## Confirmed implementation

Public API modules include `cognee/api/v1/add/add.py`, `cognify/cognify.py`, `search/search.py`, `recall/recall.py`, `remember/remember.py`, `forget/forget.py`, and `permissions`. The flow is intentionally split: add persists source data; cognify runs configured tasks; search/recall retrieve from generated structures. `cognee/tasks/storage/add_data_points.py`, `index_data_points.py`, and `index_graph_edges.py` show separate persistence/index stages.

The repository contains SQL migrations for users, tenants, datasets, pipeline status, sync operations, and graph metadata, including `cognee/alembic/versions/c946955da633_multi_tenant_support.py` and `76625596c5c3_expand_dataset_database_for_multi_user.py`. Tests cover PostgreSQL, pgvector, Neo4j, Kuzu, Turso, graph provenance, rollback recovery, orphan cleanup, deduplication, and session persistence. This confirms a broad adapter architecture, not that every backend has equal behavior.

Temporal work is present in `cognee/tasks/temporal_awareness/` and tests, including `build_graph_with_temporal_awareness.py`, `search_graph_with_temporal_awareness.py`, and `test_graph_provenance_unified_contract.py`. Agent/session state is represented in `cognee/infrastructure/session/`. The graph can be Kuzu, Neo4j, PostgreSQL-backed, or other adapters; the configuration and tests show operational breadth.

## Engineering assessment

Strong decisions: staged pipeline; explicit rollback/error tests; provenance and orphan cleanup; tenant/user migrations; adapter contracts; separate foreground/background synchronization concepts.

Weaknesses for a founder-built first product: many backends, graph/vector/relational combinations, migration surface, and task catalog create high operational complexity. A model-generated graph is not automatically an evidence-backed memory; TermyteDB should keep source evidence authoritative and derive graph indexes later.

## Reuse and rejection

Reuse task separation, provenance tests, pipeline status, and adapter contracts. Reject Cognee's breadth as V1 scope and do not require a graph database. License is Apache-2.0 (`LICENSE`, `NOTICE.md`); any reused code must preserve notices and license terms.

