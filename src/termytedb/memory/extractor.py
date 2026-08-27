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


IGNORED_KEYS = {
    "__termytedb_event_type",
    "tool_name",
    "command",
    "cmd",
    "stdout",
    "stderr",
    "exit_code",
    "error",
    "corrective_action",
    "recovery",
    "resolution",
}


def payload_text(payload: dict[str, Any], event_type: str | None = None) -> str:
    """Project a conversational payload into stable extraction text.

    The processor and rule extractor both use this function, so evidence
    offsets remain aligned with the text supplied to the extraction contract.
    """
    parts: list[str] = []

    def clean(value: str, *, strip_wrappers: bool = True) -> str:
        cleaned = value
        if strip_wrappers:
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
            for key, item in value.items():
                if key in IGNORED_KEYS:
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
    # --- Generic personal-fact fallback (conversational memory) ---
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
        re.compile(
            r"(?im)\bI\s+(?:used to|used to live in|used to work as|now live in|now work as|moved to|moved from)\b[^.!?\n]{2,200}[.!?]"
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


def _subject_key(kind: str, statement: str, body: str) -> str:
    text = f"{statement} {body}".casefold()
    if kind == "fact":
        if any(token in text for token in ("live in", "lived in", "moved to", "moved from", "move to", "move from")):
            return "fact:user location"
        if any(token in text for token in ("prefer", "favorite", "favoured", "favourite", "like ", "love ")):
            return "fact:user preference"
        if any(token in text for token in ("work as", "work in", "job", "role ", "employed")):
            return "fact:user work"
        if any(token in text for token in ("sister", "brother", "cousin", "mother", "father", "wife", "husband", "partner")):
            return "fact:user relationship"
    if kind == "decision":
        return "decision:state change"
    subject_words = body.casefold().split()[:2]
    return f"{kind}:{' '.join(subject_words)}"


def extract(payload: dict[str, Any], event_type: str | None = None) -> list[Candidate]:
    text = payload_text(payload, event_type)
    candidates: list[Candidate] = []
    for kind, pattern in RULES:
        for match in pattern.finditer(text):
            statement = match.group(0).strip()
            body = match.group(1).strip() if match.lastindex else statement
            if not body:
                continue
            subject_key = _subject_key(kind, statement, body)
            candidates.append(Candidate(kind, subject_key, statement, match.start(), match.end()))
    return sorted(candidates, key=lambda item: (item.start_offset, item.kind, item.statement))
