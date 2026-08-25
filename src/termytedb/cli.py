"""Local TermyteDB command line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .memory.consolidator import consolidate
from .runtime.engine import TermyteDB


def _db(args: argparse.Namespace) -> TermyteDB:
    return TermyteDB(args.database)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="termytedb")
    parser.add_argument("--database", default=".termytedb/memory.sqlite")
    parser.add_argument("--namespace", default="default")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    sub.add_parser("status")
    for name in ("connect",):
        command = sub.add_parser(name)
        command.add_argument("adapter", choices=("claude-code", "codex"))
    context = sub.add_parser("context")
    context.add_argument("query")
    context.add_argument("--token-budget", type=int, default=500)
    context.add_argument("--limit", type=int, default=10)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("memory_id")
    consolidate_parser = sub.add_parser("consolidate")
    consolidate_parser.add_argument("--dry-run", action="store_true")
    consolidate_parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args(argv)
    if args.command == "init":
        Path(args.database).parent.mkdir(parents=True, exist_ok=True)
        db = _db(args)
        db.repository.ensure_namespace(args.namespace)
        db.close()
        print(json.dumps({"database": str(args.database), "namespace": args.namespace, "initialized": True}))
        return 0
    db = _db(args)
    try:
        db.repository.ensure_namespace(args.namespace)
        if args.command == "status":
            print(json.dumps(db.metrics(args.namespace), indent=2))
        elif args.command == "connect":
            print(json.dumps({"adapter": args.adapter, "status": "ready", "mode": "local-event-capture"}))
        elif args.command == "context":
            result = db.context(args.namespace, args.query, args.token_budget, args.limit)
            print(result.text or json.dumps(result.model_dump(mode="json")))
        elif args.command == "inspect":
            print(json.dumps(db.repository.history(args.namespace, args.memory_id), indent=2, default=str))
        elif args.command == "consolidate":
            print(json.dumps(consolidate(db.repository, args.namespace, limit=args.limit, mode="dry-run" if args.dry_run else "apply"), indent=2))
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
