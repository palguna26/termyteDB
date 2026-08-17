# ADR 0023: Coding-agent continuation benchmark harness

## Status

Accepted

## Decision

Continuation fixtures are JSONL records containing a snapshot identifier, Agent A task, evidence trajectory, Agent B continuation task, verification description, expected outcome, repository snapshot, resulting repository state, declarative verification tests, and optional previous summary. The runner ingests each trajectory through the production engine, compares no-memory, raw-history, previous-summary, and TermyteDB context baselines, and validates the repository fixture without executing arbitrary commands. It reports completion proxy rate, context token totals, repository-fixture verification rate, elapsed time, and improvement over the previous-summary baseline.

The checked-in two-case fixture is synthetic and only proves harness behavior and declarative verification integrity. It is not a claim about real coding-agent task completion; real Agent A/Agent B trajectories and actual verification command execution remain external benchmark work.
