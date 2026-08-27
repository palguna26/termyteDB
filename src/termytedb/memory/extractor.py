from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Candidate:
    kind: str
    subject_key: str
    statement: str
    start_offset: int
    end_offset: int


EXECUTION_EVENT_TYPES = {"tool_execution", "bash_command"}
EXECUTION_FIELDS = (
    ("tool_name", "Tool"),
    ("command", "Command"),
    ("cmd", "Command"),
    ("stdout", "Stdout"),
    ("stderr", "Stderr"),
    ("exit_code", "Exit code"),
    ("error", "Error"),
    ("corrective_action", "Corrective action"),
    ("recovery", "Recovery"),
    ("resolution", "Resolution"),
)


def payload_text(payload: dict[str, Any], event_type: str | None = None) -> str:
    """Project an event payload into stable extraction text.

    Agent hooks often wrap the real request in volatile harness context. Keep
    the stored redacted payload unchanged, but prefer explicit user/assistant
    fields and remove known wrapper blocks before extraction. The processor and
    rule extractor both use this function, so evidence offsets remain aligned
    with the text supplied to the extraction contract.
    """
    parts: list[str] = []

    event_type = event_type or (str(payload.get("__termytedb_event_type")) if payload.get("__termytedb_event_type") else None)
    execution_event = (event_type or "").casefold() in EXECUTION_EVENT_TYPES

    def clean(value: str, *, strip_wrappers: bool = True) -> str:
        cleaned = value
        if strip_wrappers:
            for tag in ("additional_data", "system_reminder", "environment_context"):
                cleaned = re.sub(rf"<{tag}>.*?</{tag}>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        return cleaned.strip()

    execution_parts: list[str] = []
    seen_execution_labels: set[str] = set()

    def visit_execution(value: Any) -> None:
        if isinstance(value, dict):
            for key, label in EXECUTION_FIELDS:
                item = value.get(key)
                if item is None or label in seen_execution_labels:
                    continue
                rendered = clean(str(item), strip_wrappers=False)
                if rendered:
                    execution_parts.append(f"{label}: {rendered}")
                    seen_execution_labels.add(label)
            for item in value.values():
                visit_execution(item)
        elif isinstance(value, list):
            for item in value:
                visit_execution(item)

    visit_execution(payload)

    def visit_message(value: Any) -> None:
        if not isinstance(value, dict):
            visit(value)
            return
        role = value.get("role")
        if isinstance(role, str) and role.casefold() == "system":
            return
        content = value.get("content", value.get("text"))
        if isinstance(content, str):
            text = clean(content, strip_wrappers=not execution_event)
            if text:
                parts.append(text)
        elif content is not None:
            visit(content)

    def visit(value: Any) -> None:
        if isinstance(value, str):
            text = clean(value, strip_wrappers=not execution_event)
            if text:
                parts.append(text)
        elif isinstance(value, dict):
            for key, item in value.items():
                if key == "__termytedb_event_type":
                    continue
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    explicit = False
    for key in ("user_query", "user_message", "query"):
        value = payload.get(key)
        if isinstance(value, str) and clean(value):
            parts.append(clean(value))
            explicit = True
    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in messages:
            visit_message(message)
        explicit = explicit or bool(parts)
    if not explicit and not execution_parts:
        visit(payload)
    for item in execution_parts:
        if item.casefold() not in {part.casefold() for part in parts}:
            parts.append(item)
    return "\n".join(parts)


RULES = (
    (
        "fact",
        re.compile(
            r"(?im)^(?:the|this|service|repository|version|file|branch)\s+[^.!?\n]{1,160}\s+(?:is|runs on|uses|requires|contains)\s+[^.!?\n]{1,240}[.!?]"
        ),
    ),
    # --- Generic personal-fact fallback (ordinary agent conversations) ---
    # Captures first-person and possessive facts that the original service-oriented
    # pattern misses, e.g. "I graduated with a degree in Business Administration."
    # or "My favorite color is blue." Keeps evidence offsets stable via payload_text.
    (
        "fact",
        re.compile(
            r"(?im)\bI\s+(?:graduated\s+with|have|has|had|am|was|were|live\s+in|lived\s+in|work\s+as|work\s+in|own|owns|like|liked|love|loved|prefer|preferred)\b[^.!?\n]{2,200}[.!?]"
        ),
    ),
    (
        "fact",
        re.compile(r"(?im)\bMy\s+[^.!?\n]{2,120}\s+is\s+[^.!?\n]{2,160}[.!?]"),
    ),
    (
        "fact",
        re.compile(r"(?im)\bI\s+am\s+[^.!?\n]{2,120}[.!?]"),
    ),
    (
        "decision",
        re.compile(r"(?i)\b(?:decision|decided|choose|chosen)\s*[:\-]\s*(.+?)(?:[.!?]|$)"),
    ),
    ("decision", re.compile(r"(?i)\bwe decided to\s+(.+?)(?:[.!?]|$)")),
    ("failure", re.compile(r"(?i)\b(?:failure|failed|error)\s*[:\-]\s*(.+?)(?:[.!?]|$)")),
    ("outcome", re.compile(r"(?i)\boutcome\s*[:\-]\s*(.+?)(?:[.!?]|$)")),
    ("constraint", re.compile(r"(?i)\bconstraint\s*[:\-]\s*(.+?)(?:[.!?]|$)")),
    ("procedure", re.compile(r"(?i)\bprocedure\s*[:\-]\s*(.+?)(?:[.!?]|$)")),
    ("attempt", re.compile(r"(?i)\battempt\s*[:\-]\s*(.+?)(?:[.!?]|$)")),
    ("task_state", re.compile(r"(?i)\btask\s*[:\-]\s*(.+?)(?:[.!?]|$)")),
    ("question", re.compile(r"(?i)\bquestion\s*[:\-]\s*(.+?)(?:[.!?]|$)")),
    ("correction", re.compile(r"(?i)\b(?:correction|corrected)\s*[:\-]\s*(.+?)(?:[.!?]|$)")),
)


def extract(payload: dict[str, Any], event_type: str | None = None) -> list[Candidate]:
    text = payload_text(payload, event_type)
    candidates: list[Candidate] = []
    for kind, pattern in RULES:
        for match in pattern.finditer(text):
            statement = match.group(0).strip()
            body = match.group(1).strip() if match.lastindex else statement
            if not body:
                continue
            subject_words = body.casefold().split()[:2]
            subject_key = f"{kind}:{' '.join(subject_words)}"
            candidates.append(Candidate(kind, subject_key, statement, match.start(), match.end()))
    return sorted(candidates, key=lambda item: (item.start_offset, item.kind, item.statement))
