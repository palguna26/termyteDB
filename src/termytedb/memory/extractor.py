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


def payload_text(payload: dict[str, Any]) -> str:
    """Project an event payload into stable extraction text.

    Agent hooks often wrap the real request in volatile harness context. Keep
    the stored redacted payload unchanged, but prefer explicit user/assistant
    fields and remove known wrapper blocks before extraction. The processor and
    rule extractor both use this function, so evidence offsets remain aligned
    with the text supplied to the extraction contract.
    """
    parts: list[str] = []

    def clean(value: str) -> str:
        cleaned = value
        for tag in ("additional_data", "system_reminder", "environment_context"):
            cleaned = re.sub(rf"<{tag}>.*?</{tag}>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        return cleaned.strip()

    def visit_message(value: Any) -> None:
        if not isinstance(value, dict):
            visit(value)
            return
        role = value.get("role")
        if isinstance(role, str) and role.casefold() == "system":
            return
        content = value.get("content", value.get("text"))
        if isinstance(content, str):
            text = clean(content)
            if text:
                parts.append(text)
        elif content is not None:
            visit(content)

    def visit(value: Any) -> None:
        if isinstance(value, str):
            text = clean(value)
            if text:
                parts.append(text)
        elif isinstance(value, dict):
            for item in value.values():
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
    if not explicit:
        visit(payload)
    return "\n".join(parts)


RULES = (
    (
        "fact",
        re.compile(
            r"(?im)^(?:the|this|service|repository|version|file|branch)\s+[^.!?\n]{1,160}\s+(?:is|runs on|uses|requires|contains)\s+[^.!?\n]{1,240}[.!?]"
        ),
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


def extract(payload: dict[str, Any]) -> list[Candidate]:
    text = payload_text(payload)
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
