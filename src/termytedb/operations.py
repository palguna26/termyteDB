from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .engine import TermyteDB
from .evaluation import run_performance_benchmark
from .integrity import check_database, repair_fts


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TermyteDB local database operations")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="initialize a database")
    init.add_argument("--database", required=True, type=Path)

    export = subparsers.add_parser("export", help="export one namespace as JSON")
    export.add_argument("--database", required=True, type=Path)
    export.add_argument("--namespace", required=True)
    export.add_argument("--output", required=True, type=Path)

    import_command = subparsers.add_parser("import", help="import a namespace JSON document")
    import_command.add_argument("--database", required=True, type=Path)
    import_command.add_argument("--namespace", required=True)
    import_command.add_argument("--input", required=True, type=Path)

    backup = subparsers.add_parser("backup", help="create a SQLite online backup")
    backup.add_argument("--database", required=True, type=Path)
    backup.add_argument("--output", required=True, type=Path)

    integrity = subparsers.add_parser("integrity", help="check or repair derived indexes")
    integrity.add_argument("--database", required=True, type=Path)
    integrity.add_argument("--repair-fts", action="store_true")

    benchmark = subparsers.add_parser("benchmark", help="run the local production benchmark")
    benchmark.add_argument("--events", type=int, default=100)

    args = parser.parse_args(arguments)
    if args.command == "benchmark":
        print(json.dumps(run_performance_benchmark(args.events), sort_keys=True))
        return 0
    if args.command == "init":
        db = TermyteDB(args.database)
        db.close()
        return 0
    if args.command == "export":
        db = TermyteDB(args.database)
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(db.export_namespace(args.namespace), indent=2, sort_keys=True), encoding="utf-8")
        finally:
            db.close()
        return 0
    if args.command == "import":
        db = TermyteDB(args.database)
        try:
            result = db.import_namespace(json.loads(args.input.read_text(encoding="utf-8")), args.namespace)
        finally:
            db.close()
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "backup":
        db = TermyteDB(args.database)
        try:
            db.backup(args.output)
        finally:
            db.close()
        return 0
    db = TermyteDB(args.database)
    try:
        if args.repair_fts:
            repair_fts(db.database)
        report = check_database(db.database)
        print(json.dumps(report.__dict__, sort_keys=True))
        return 0 if report.ok else 1
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
