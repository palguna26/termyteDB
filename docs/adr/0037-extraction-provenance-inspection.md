# ADR 0037: Extraction provenance inspection

## Decision

Expose namespace-scoped, paginated HTTP read paths for extraction runs and
extraction decisions. These paths return the persisted provider metadata,
validation status, rejection reason, reconciliation action, and linked IDs.

## Reason

Extraction and reconciliation are evidence-backed only when operators can
inspect why a candidate was accepted or rejected. The API uses the same
namespace predicate as the engine and keeps the existing bounded pagination
limits.
