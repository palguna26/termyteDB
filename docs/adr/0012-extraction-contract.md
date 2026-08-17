# ADR 0012: Strict evidence-constrained extraction contract

Status: accepted

Milestone 2 treats a model as an untrusted proposal generator. `ExtractionCandidate` is a closed Pydantic schema (`extraction-v1`) with fixed memory kinds, bounded statements and evidence, confidence, durability, validity, and reconciliation intent. The processor validates every field and exact span before any memory transaction.

Unknown fields, unsupported claims, cross-input event IDs, secret-bearing values, and invalid spans are rejected. Rejected decisions keep only redacted structured fields and a machine-readable reason. Raw provider output is not persisted.

Free-form JSON and model-only evidence checking were rejected because exact span existence must remain deterministic.
