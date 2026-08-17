# ADR 0023: Coding-agent continuation benchmark harness

## Status

Accepted

## Decision

Continuation fixtures are JSONL records containing a snapshot identifier, Agent A task, evidence trajectory, Agent B continuation task, verification description, expected outcome, and optional previous summary. The runner ingests each trajectory through the production engine, then compares no-memory, raw-history, previous-summary, and TermyteDB context baselines. It reports completion proxy rate, context token totals, elapsed time, and improvement over the previous-summary baseline.

The checked-in two-case fixture is synthetic and only proves harness behavior. It is not a claim about real coding-agent task completion; real repository snapshots and verification commands must be added before release claims.
