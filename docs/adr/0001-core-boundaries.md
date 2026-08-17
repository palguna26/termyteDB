# ADR 0001: Core boundaries

Status: accepted.

TermyteDB engine and HTTP service define framework-neutral events, memories, retrieval, and diagnostics. CLI, SDK, and adapters are clients. This follows the need for integrations to vary while the memory lifecycle remains stable; Tencent's direct adapters show why coupling must be avoided. Rejected: embedding one agent's session hooks in the engine.

