# ADR 0013: Deterministic reconciliation actions

Status: accepted

Validated proposals use one of INSERT, REINFORCE, UPDATE, SUPERSEDE, DISPUTE, or IGNORE. Identical active claims reinforce. Conflicting claims are disputed by default. UPDATE and SUPERSEDE require explicit correction or replacement language; proposal intent alone cannot close current truth. Every action is recorded with its run, fingerprint, and resulting IDs.

The deterministic rule extractor retains Milestone 1 compatibility behavior. The model-provider path uses the stricter reconciliation policy.
