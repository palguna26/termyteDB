# ADR 0007: Milestone 1 uses flat HTTP paths

Status: accepted for Milestone 1.

The architecture example shows namespace-bearing paths such as `/v1/namespaces/{namespace_id}/events`, while the Milestone 1 request explicitly requires `/v1/events`, `/v1/process`, `/v1/search`, `/v1/context`, and `/v1/memories/{id}`. The implementation follows the milestone contract and requires `namespace_id` in every request body or as a required query parameter for memory lookup. Repository methods still require the namespace as a separate argument and apply it inside every SQL query.

This keeps the engine boundary stable while leaving URL nesting as a service-version decision. No namespace is inferred from a prompt or memory ID.

