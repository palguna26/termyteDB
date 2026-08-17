from time import perf_counter

from termytedb import TermyteDB
from termytedb.provider import ProviderResult
from termytedb.schemas import ExtractionResponse


class RecordingProvider:
    name = "recording"
    model = "recording-v1"

    def __init__(self):
        self.timeout_seconds = None
        self.cancelled = None

    def extract(self, request, timeout_seconds=30.0, cancellation=None):
        del request
        self.timeout_seconds = timeout_seconds
        self.cancelled = cancellation is not None
        response = ExtractionResponse(schema_version="extraction-v1", prompt_version="p", candidates=[])
        return ProviderResult(response, self.name, self.model, "p", "hash", None, None, int(perf_counter() * 0))


def test_processing_passes_timeout_and_cancellation_to_provider(tmp_path):
    provider = RecordingProvider()
    db = TermyteDB(tmp_path / "provider-control.sqlite", extraction_provider=provider)
    db.ingest({"namespace_id": "provider", "idempotency_key": "one", "type": "note", "payload": {"text": "unstructured"}})
    db.process_with_timeout("provider", timeout_seconds=5)
    assert provider.timeout_seconds is not None
    assert 0 < provider.timeout_seconds <= 5
    assert provider.cancelled is True
    db.close()
