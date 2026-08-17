# Evaluation design

The harness invokes the production ingest, process, search, and context interfaces. Fixtures are versioned JSONL with namespace, events, expected memories, evidence spans, valid intervals, conflicts, and expected retrieval labels.

Component gates measure extraction precision/recall, attribution accuracy, duplicate detection, ADD/UPDATE/CONTRADICT classification, temporal interval accuracy, and unsupported-memory rejection. Retrieval gates measure Recall@k, Precision@k, NDCG, stale rejection, zero cross-namespace leakage, abstention calibration, and token-normalized usefulness.

The continuation benchmark packages a repository snapshot, Agent A task/trajectory/decisions/discoveries/failures, resulting repository state, Agent B continuation task, and verification tests. Compare no memory, raw transcript, summary, vector-only, Mem0 where feasible, Graphiti where feasible, TermyteDB, and oracle context. Record completion, tests, time, calls, tokens, repeated mistakes, incorrect assumptions, exploration, and context use.

LongMemEval-s gets an adapter that transforms each item into the same event/evidence pipeline, pins model/embedding versions, records seeds/configuration, and emits raw predictions plus metrics. No benchmark-specific shortcut may bypass production processing.

Milestone 2 adds `tests/fixtures/extraction_cases.jsonl` with 50 labelled cases and `python -m termytedb.evaluation <fixture>`. The command reports a deterministic rule-only baseline. Reconciliation and temporal scores remain zero until labelled state transitions are run through the model-provider harness.

The retrieval evaluator accepts JSONL cases containing `query`, `evidence`, and `expected_statement`. `python -m termytedb.evaluation <fixture> --retrieval` writes Recall@k, MRR, NDCG@k, elapsed time, and case count after using the production ingest, process, and search path. The checked-in four-case smoke fixture measured Recall@5=1.0, MRR=1.0, and NDCG@5=1.0 locally; this is a regression fixture, not a general quality claim.

Reconciliation fixtures contain ordered event text and expected actions. `python -m termytedb.evaluation <fixture> --reconciliation` runs the production ingest and processing path and reports reconciliation accuracy. The checked-in fixture covers INSERT, REINFORCE, and DISPUTE; it is a deterministic regression fixture, not a broad quality claim.

Temporal fixtures provide labelled validity intervals. `python -m termytedb.evaluation <fixture> --temporal` runs the real provider, reconciliation, and retrieval path and reports stale-memory rejection plus historical temporal-state accuracy.

Continuation fixtures run with `python -m termytedb.evaluation <fixture> --continuation`. The checked-in synthetic fixture measured no-memory completion `0.0`, previous-summary completion `0.0`, and TermyteDB completion `1.0`; raw history also reached `1.0`. This demonstrates the harness and production path only, not a general agent-quality claim.

LongMemEval-shaped fixtures run with `python -m termytedb.evaluation <fixture> --longmemeval`. The adapter emits frozen configuration, prompt hash, raw predictions, abstention, token, latency, and accuracy fields. The checked-in two-item fixture reached accuracy `1.0`; it is synthetic and is not an external LongMemEval-s result.

`run_performance_benchmark(n)` measures local operation latency, job throughput, bounded p95-style search/context samples, on-disk SQLite size, and restart recovery. A 10-event local run measured batch ingest `12.802 ms` (`781.15 events/s`), processing `43.455 ms`, search `0.768 ms`, context `0.332 ms`, and restart search `0.725 ms` on the developer machine. These are one smoke run, not release capacity targets.

The same benchmark also runs bounded concurrent ingestion across four namespaces and reports `concurrent_namespace_ms` and the recovered job count. This is a local contention smoke measurement, not a capacity or p95 claim.
