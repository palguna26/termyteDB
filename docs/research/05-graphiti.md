# Graphiti source review

## Confirmed implementation

`graphiti_core/graphiti.py` is the main orchestrator. It accepts episodes and triplets, extracts nodes and edges through `graphiti_core/utils/maintenance/combined_extraction.py`, resolves pointers, deduplicates nodes/edges, and writes episode/entity/community structures. `graphiti_core/nodes.py`, `edges.py`, and `graphiti_types.py` define episodic nodes, entity nodes, community nodes, facts, and temporal fields.

`graphiti_core/driver/operations/search_ops.py` and backend-specific search operations implement multiple retrieval forms: node/edge full-text and vector search, graph traversal/BFS-style expansion, and reciprocal-rank-fusion/shortlisting utilities. Drivers include Neo4j, FalkorDB, Kuzu, and Neptune. `group_id` is validated and used to separate graph groups. Tests cover search security, node-label security, edge cross-encoder RRF shortlist, BFS query shape, entity exclusion, and multiple group IDs.

Temporal edge handling is explicit in the edge models and maintenance operations: facts can have valid/invalid timestamps and updates can invalidate prior facts. This is a strong fit for changing project decisions. However, the graph database is authoritative for the system's graph objects, and extraction/deduplication requires substantial LLM, embedding, and database infrastructure.

## Engineering assessment

Strong decisions: episode provenance as graph structure; temporal fact invalidation; group isolation; backend driver interface; hybrid retrieval; security-focused query tests; explicit deduplication and entity resolution.

Weaknesses: graph backend operations and schema complexity; model-generated entities can create false links; graph traversal can return plausible but weakly supported context; local and hosted deployments carry different databases and tuning requirements.

## Reuse and rejection

Reuse temporal fact fields, episode-to-fact provenance, group filtering, hybrid candidate generation, and security tests. Reject Neo4j/FalkorDB as a V1 dependency; represent relationships in relational tables until coding-agent benchmarks show graph recall materially beats simpler retrieval. License is Apache-2.0 (`LICENSE`); preserve notices for reused code.

