# Risk register

1. Model invents or misattributes a memory. Mitigate with evidence-span validation, quarantine, and attribution gates.
2. Contradictory project facts produce wrong context. Mitigate with append-only versions, valid-time filters, conflict diagnostics, and labelled tests.
3. Namespace leakage. Mitigate with scope columns, repository methods that require scope, SQL tests, and adversarial cross-tenant fixtures.
4. Async jobs lose or duplicate work. Mitigate with idempotency keys, leases, checkpoints, retries, dead letters, and replay tests.
5. Retrieval looks relevant but wastes context. Mitigate with bounded context, diversity, abstention, token-normalized evaluation, and coding continuation tests.
6. Provider costs/latency dominate. Mitigate with sync persistence, async extraction, local deterministic stages, cached embeddings, and per-job telemetry.
7. Graph scope expands the product. Mitigate with relational relationship rows and an explicit graph ablation gate.
8. Deletion conflicts with provenance/audit requirements. Define retention classes and cryptographic tombstones before hosted launch.

