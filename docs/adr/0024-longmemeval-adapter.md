# ADR 0024: LongMemEval-s production-path adapter

## Status

Accepted

## Decision

The LongMemEval-shaped adapter accepts items with evidence, question, expected answer, and stable ID. Each item is isolated in a namespace and uses production ingestion, processing, and context retrieval. Output freezes dataset revision, extraction/embedding/reranker/answer configuration, retrieval weights, top-k, token budget, prompt hash, raw predictions, abstention, token count, accuracy, and latency.

The checked-in fixture is local and synthetic. It proves reproducibility and path wiring only; it is not the external LongMemEval-s dataset and must not be reported as its score.
