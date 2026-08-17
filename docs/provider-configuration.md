# Provider configuration

Rule extraction is the default and needs no account or network. The deterministic fake provider is available to tests. For a real model, construct `HttpExtractionProvider` with an endpoint and model, or set:

```text
TERMYTEDB_EXTRACTION_URL=https://your-provider.example/extract
TERMYTEDB_EXTRACTION_MODEL=your-model
TERMYTEDB_EXTRACTION_API_KEY=your-secret
```

The endpoint receives JSON containing `model`, a delimited evidence prompt, and `schema: extraction-v1`. It must return an `extraction-v1` object, optionally wrapped as a JSON string in an `output` field. Credentials are never included in provider errors.
