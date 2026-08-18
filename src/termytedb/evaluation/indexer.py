"""Local LongMemEval atom and dense-vector indexer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..retrieval.embedding import FastEmbedProvider
from ..storage.db import Database
from .longmemeval_extraction import L1Atom, index_atom_embeddings, insert_atoms


def index_sessions(database: Database, sessions_data: list[dict[str, Any]]) -> tuple[int, int]:
    atoms: list[L1Atom] = []
    for question in sessions_data:
        session_ids = question.get("haystack_session_ids", [])
        dates = question.get("haystack_dates", [])
        for session_index, session in enumerate(question.get("haystack_sessions", [])):
            session_id = str(session_ids[session_index] if session_index < len(session_ids) else f"session-{session_index}")
            timestamp = str(dates[session_index]) if session_index < len(dates) else None
            for turn_index, turn in enumerate(session):
                if not isinstance(turn, dict) or not turn.get("content"):
                    continue
                atom_id = f"{question.get('question_id', 'dataset')}:{session_id}:{turn_index}"
                atoms.append(L1Atom(atom_id, session_id, str(turn["content"]), timestamp, str(turn.get("role", "user"))))
    inserted = insert_atoms(database, atoms)
    indexed = index_atom_embeddings(database, FastEmbedProvider())
    return inserted, indexed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--database", type=Path, default=Path("termyte_dryrun.sqlite"))
    args = parser.parse_args()
    data = json.loads(args.data_path.read_text(encoding="utf-8"))
    db = Database(args.database)
    try:
        inserted, indexed = index_sessions(db, data)
        print(f"Inserted atoms: {inserted}; dense vectors: {indexed}")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
