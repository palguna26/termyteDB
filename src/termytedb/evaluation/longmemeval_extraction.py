from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

from ..memory.provider import ProviderResult
from ..storage.db import Database

CANDIDATE_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
]


@dataclass(frozen=True)
class L1Atom:
    atom_id: str
    session_id: str
    fact: str
    timestamp: str | None
    source_role: str
    namespace_id: str | None = None


class AtomProvider(Protocol):
    def extract_atoms(self, session_id: str, messages: list[dict[str, Any]]) -> ProviderResult: ...


def atom_prompt(session_id: str, messages: list[dict[str, Any]]) -> str:
    """Build the strict, low-token extraction contract used by Gemini adapters."""
    transcript = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return (
        "Extract durable facts from this session. Return JSON only: "
        '{"atoms":[{"fact":"third-person standalone claim",'
        '"timestamp":"ISO-8601 or null","source_role":"user|assistant"}]}.'
        " Emit one independent fact per atom. Preserve explicit dates and resolve relative dates "
        "from the session metadata when possible. Do not infer unsupported facts. "
        f"session_id={session_id}; messages={transcript}"
    )


def insert_atoms(db: Database, atoms: list[L1Atom], *, created_at: str | None = None) -> int:
    now = created_at or datetime.now(UTC).isoformat()
    with db.connection:
        for atom in atoms:
            if atom.source_role not in {"user", "assistant"}:
                raise ValueError("source_role must be user or assistant")
            db.execute(
                """INSERT OR IGNORE INTO atoms
                   (atom_id, session_id, fact, timestamp, source_role, created_at, namespace_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (atom.atom_id, atom.session_id, atom.fact.strip(), atom.timestamp, atom.source_role, now, atom.namespace_id),
            )
    return len(atoms)


def index_atom_embeddings(db: Database, provider: Any, *, batch_size: int = 64) -> int:
    """Index current atoms with an injectable dense provider."""
    rows = db.execute(
        """SELECT a.atom_id, a.fact FROM atoms a
           LEFT JOIN atom_embeddings e ON e.atom_id=a.atom_id AND e.provider=?
           WHERE e.atom_id IS NULL ORDER BY a.rowid""",
        (provider.name,),
    ).fetchall()
    count = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        values = [str(row["fact"]) for row in batch]
        if hasattr(getattr(provider, "model", None), "embed"):
            vectors = list(provider.model.embed(values, batch_size=batch_size))
        elif hasattr(provider, "embed_many"):
            vectors = provider.embed_many(values)
        else:
            vectors = [provider.embed(value) for value in values]
        with db.connection:
            for row, vector in zip(batch, vectors, strict=True):
                vector_values = [float(item) for item in vector]
                import array
                blob = array.array("f", vector_values).tobytes()
                db.execute(
                    "INSERT OR REPLACE INTO atom_embeddings(atom_id, provider, dimensions, vector) VALUES (?, ?, ?, ?)",
                    (row["atom_id"], provider.name, len(vector_values), blob),
                )
                count += 1
    return count


def atoms_from_provider_result(session_id: str, result: ProviderResult) -> list[L1Atom]:
    atoms: list[L1Atom] = []
    for candidate in result.response.candidates:
        timestamp = candidate.timestamp or candidate.valid_from
        atoms.append(
            L1Atom(
                atom_id=str(uuid4()),
                session_id=session_id,
                fact=candidate.statement,
                timestamp=timestamp.astimezone(UTC).isoformat() if timestamp else None,
                source_role=candidate.source_role,
            )
        )
    return atoms


def extract_session(db: Database, provider: AtomProvider, session_id: str, messages: list[dict[str, Any]]) -> list[L1Atom]:
    """Extract and persist one session; provider calls remain injectable for zero-spend tests."""
    result = provider.extract_atoms(session_id, messages)
    atoms = atoms_from_provider_result(session_id, result)
    insert_atoms(db, atoms)
    return atoms
