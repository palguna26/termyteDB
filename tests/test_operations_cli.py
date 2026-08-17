import json

from termytedb import TermyteDB
from termytedb.operations import main


def test_operations_cli_supports_init_export_import_backup_and_integrity(tmp_path):
    source = tmp_path / "source.sqlite"
    export_file = tmp_path / "export.json"
    backup = tmp_path / "backup.sqlite"
    imported = tmp_path / "imported.sqlite"
    assert main(["init", "--database", str(source)]) == 0
    db = TermyteDB(source)
    db.ingest({"namespace_id": "ops", "idempotency_key": "one", "type": "note", "payload": {"text": "Decision: use SQLite."}})
    db.process("ops")
    db.close()
    assert main(["export", "--database", str(source), "--namespace", "ops", "--output", str(export_file)]) == 0
    assert json.loads(export_file.read_text(encoding="utf-8"))["events"]
    assert main(["import", "--database", str(imported), "--namespace", "ops", "--input", str(export_file)]) == 0
    assert main(["backup", "--database", str(source), "--output", str(backup)]) == 0
    assert main(["integrity", "--database", str(backup)]) == 0
    restored = TermyteDB(backup)
    assert restored.search("ops", "SQLite")
    restored.close()


def test_operations_cli_benchmark_reports_metrics(capsys):
    assert main(["benchmark", "--events", "2"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["event_count"] == 2
    assert output["concurrent_namespace_count"] == 4
