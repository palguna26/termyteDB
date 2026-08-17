# Open questions for experiments

- Which event types and evidence span formats provide the best coding-agent attribution?
- Does vector retrieval beat FTS for repository decisions after code/token normalization?
- What score threshold gives useful abstention without harming continuation recall?
- Are entity links needed for V1 coding tasks, or do scoped subject keys suffice?
- Does a relationship index improve Recall@20 enough to justify maintenance?
- Which local embedding/model combination meets latency and quality gates on the target hardware?
- What retention and deletion policy is acceptable for hosted customer evidence?
- How should branch scope inherit or isolate project memories?
- What is the smallest LongMemEval-s adapter that preserves benchmark comparability?

These require fixtures and measurements, not more architecture prose.

