# Simple LongMemEval OpenRouter command

## Goal

Run the complete TermyteDB LongMemEval-S pipeline with one command:

```powershell
python benchmarks/longmemeval/run_end_to_end.py --openrouter
```

The only required user configuration is `OPENROUTER_API_KEY` in the repository-root `.env` file.

## Defaults

- Dataset: `benchmarks/longmemeval/longmemeval_s_cleaned.json`
- Extraction provider: OpenRouter
- Model: `inclusionai/ling-2.6-flash`
- Workers: 4
- Process batch: 1 job per worker
- Retrieval: top 5
- Artifacts: timestamped SQLite database and JSON result under `benchmarks/longmemeval/runs/`

## Behavior

`--openrouter` selects the OpenRouter extraction path and applies the defaults above. Existing detailed options remain available and override defaults. The script loads `.env` before checking credentials. A missing key produces a short setup error before dataset ingestion.

Progress output includes successful, failed, dead-lettered, and pending job counts. The final JSON records the extraction model. Database and result paths are printed before work starts.

## Safety and errors

- Never print or store the API key.
- Refuse to start without `OPENROUTER_API_KEY`.
- Require an OpenRouter endpoint that supports strict structured output.
- Preserve timestamped databases so failed runs can be inspected or resumed.
- Stop the run early when repeated provider failures show the selected route is unhealthy.

## Verification

- CLI parsing test for the one-command defaults.
- Missing-key test that fails before ingestion.
- Override test for dataset, model, workers, database, and output.
- Provider request test for strict structured-output routing.
- Small live smoke test using one dataset session before starting the full paid run.

