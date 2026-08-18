from time import perf_counter

from termytedb import TermyteDB
from termytedb.provider import ProviderResult
from termytedb.schemas import ExtractionResponse


class RecordingProvider:
    name = "recording"
    model = "recording-v1"

    def __init__(self, input_tokens=None, output_tokens=None):
        self.timeout_seconds = None
        self.cancelled = None
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def extract(self, request, timeout_seconds=30.0, cancellation=None):
        del request
        self.timeout_seconds = timeout_seconds
        self.cancelled = cancellation is not None
        response = ExtractionResponse(schema_version="extraction-v1", prompt_version="p", candidates=[])
        return ProviderResult(response, self.name, self.model, "p", "hash", self.input_tokens, self.output_tokens, int(perf_counter() * 0))


def test_processing_passes_timeout_and_cancellation_to_provider(tmp_path):
    provider = RecordingProvider()
    db = TermyteDB(tmp_path / "provider-control.sqlite", extraction_provider=provider)
    db.ingest({"namespace_id": "provider", "idempotency_key": "one", "type": "note", "payload": {"text": "unstructured"}})
    db.process_with_timeout("provider", timeout_seconds=5)
    assert provider.timeout_seconds is not None
    assert 0 < provider.timeout_seconds <= 5
    assert provider.cancelled is True
    db.close()


def test_processing_persists_configured_estimated_provider_cost(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMYTEDB_INPUT_COST_PER_1K_USD", "1.0")
    monkeypatch.setenv("TERMYTEDB_OUTPUT_COST_PER_1K_USD", "2.0")
    provider = RecordingProvider(10, 5)
    db = TermyteDB(tmp_path / "provider-cost.sqlite", extraction_provider=provider)
    db.ingest({"namespace_id": "cost", "idempotency_key": "one", "type": "note", "payload": {"text": "unstructured"}})
    db.process("cost")
    cost = db.database.execute("SELECT estimated_cost_usd FROM extraction_runs WHERE namespace_id='cost'").fetchone()[0]
    assert cost == 0.02
    assert db.metrics("cost")["estimated_extraction_cost_usd"] == cost
    db.close()
