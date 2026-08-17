# Evaluation design

The harness invokes the production ingest, process, search, and context interfaces. Fixtures are versioned JSONL with namespace, events, expected memories, evidence spans, valid intervals, conflicts, and expected retrieval labels.

Component gates measure extraction precision/recall, attribution accuracy, duplicate detection, ADD/UPDATE/CONTRADICT classification, temporal interval accuracy, and unsupported-memory rejection. Retrieval gates measure Recall@k, Precision@k, NDCG, stale rejection, zero cross-namespace leakage, abstention calibration, and token-normalized usefulness.

The continuation benchmark packages a repository snapshot, Agent A task/trajectory/decisions/discoveries/failures, resulting repository state, Agent B continuation task, and verification tests. Compare no memory, raw transcript, summary, vector-only, Mem0 where feasible, Graphiti where feasible, TermyteDB, and oracle context. Record completion, tests, time, calls, tokens, repeated mistakes, incorrect assumptions, exploration, and context use.

LongMemEval-s gets an adapter that transforms each item into the same event/evidence pipeline, pins model/embedding versions, records seeds/configuration, and emits raw predictions plus metrics. No benchmark-specific shortcut may bypass production processing.

