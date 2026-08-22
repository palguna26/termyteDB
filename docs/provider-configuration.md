# Provider configuration

Rule extraction is the default and needs no account or network. The deterministic fake provider is available to tests. For a real model, construct `HttpExtractionProvider` with an endpoint and model, or set:

```text
TERMYTEDB_EXTRACTION_URL=https://your-provider.example/extract
TERMYTEDB_EXTRACTION_MODEL=your-model
TERMYTEDB_EXTRACTION_API_KEY=your-secret
```

The endpoint receives JSON containing `model`, a delimited evidence prompt, and `schema: extraction-v1`. It must return an `extraction-v1` object, optionally wrapped as a JSON string in an `output` field. Credentials are never included in provider errors.

For OpenRouter chat-completion extraction, set `OPENROUTER_API_KEY` and
`TERMYTEDB_EXTRACTION_MODEL`, or pass the LongMemEval benchmark's
`--extraction-model` option. OpenRouter routing only uses endpoints that support
every requested parameter, including the strict extraction JSON schema.
The LongMemEval end-to-end benchmark automatically loads these values from a
repository-root `.env` file.

With the key configured, the complete paid benchmark uses the bundled dataset,
`inclusionai/ling-2.6-flash`, four workers, and timestamped run artifacts:

```powershell
python benchmarks/longmemeval/run_end_to_end.py --openrouter
```

## OpenAI-compatible embeddings

The local FastEmbed provider remains the default. To use OpenRouter or another OpenAI-compatible embeddings endpoint, set `TERMYTEDB_EMBEDDING_PROVIDER=openrouter`, `OPENROUTER_API_KEY`, `TERMYTEDB_EMBEDDING_MODEL`, and `TERMYTEDB_EMBEDDING_DIMENSIONS`. The default endpoint is `https://openrouter.ai/api/v1`; set `TERMYTEDB_EMBEDDING_BASE_URL` for another endpoint. The provider sends batched `POST /embeddings` requests, retries transient failures, validates vector dimensions, and never logs the API key.

Optional cost telemetry uses `TERMYTEDB_INPUT_COST_PER_1K_USD` and `TERMYTEDB_OUTPUT_COST_PER_1K_USD`. Both must be set for an estimated cost to be persisted; missing or invalid values produce no estimate.

The HTTP entry point accepts `--extraction-url`, `--extraction-model`, and
`--rate-limit-per-minute`. The URL and model can also come from the environment
variables above. If no extraction URL is configured, the service uses the
offline rule path. Namespace authorization remains an application-factory
callback so hosted deployments can bind it to their identity system.
