# ADR 0002: Evidence-first memory

Status: accepted.

Immutable evidence is authoritative; every generated memory version needs evidence references. This combines Tencent's append-first ingestion with Cognee/Graphiti provenance and addresses Mem0's vector-first reconciliation risk. Rejected: storing only a summary or vector and trying to reconstruct evidence later.

