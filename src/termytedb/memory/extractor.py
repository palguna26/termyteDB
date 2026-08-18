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
    parts: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

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
