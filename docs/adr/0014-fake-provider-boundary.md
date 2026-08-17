# ADR 0014: Small offline extraction provider boundary

Status: accepted

Milestone 2 adds one structured extraction provider protocol and an offline fake provider. The default engine has no model provider and continues to run the rule-only path. Provider results carry provider/model identity, schema and prompt versions, response hash, token counts, and latency. No live provider or credential path is added in this milestone.

A provider registry, embeddings, and reranking were rejected as unnecessary Milestone 2 scope.
