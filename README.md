# DeepSearch

DeepSearch is an autonomous research worker. Given a research mission (an
objective, success criteria, deliverables and optional source limits), it
searches the public web, reads the sources, writes a cited Markdown report,
records citation checks, and returns the report together with a structured result
protocol and quality telemetry.

It is the research engine behind [TraceLab](https://github.com/kneelinghorse/TraceLab),
which authors missions and stores the results. DeepSearch can also run a
mission from a local YAML file.

## How it works

Each mission runs as one pipeline:

1. **Deterministic retrieval.** Hybrid BM25 plus dense embeddings
   (`BAAI/bge-small-en-v1.5`), fused with reciprocal-rank fusion and weighted by
   source authority (`deepsearch/agent/tools/rerank.py`).
2. **Research agent.** A [smolagents](https://github.com/huggingface/smolagents)
   `CodeAgent` with a 40-step budget drives seven tools: web search, source
   fetch, URL liveness, scratchpad read/write, scholarly search and a
   completeness check (`deepsearch/agent/codeagent_runner.py`).
3. **Critique.** Citations are extracted with exact character offsets, titles
   resolved and URLs probed for liveness. A critique pass then reviews claims
   against the collected evidence while preserving usable paid artifacts
   (`deepsearch/agent/critique.py`).
4. **Output.** The report, citations and protocol are rendered and persisted
   (`deepsearch/agent/codeagent_finalize_node.py`).

The model provider is configurable: DeepSeek (default `deepseek-flash`) or an
OpenAI-compatible endpoint. See [docs/model_configuration.md](docs/model_configuration.md).

## Run a mission locally

Requires Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
cp .env.example .env   # fill in keys only when ready to run research
```

For research (not the credential-free validation below), set a model key (`DEEPSEEK_API_KEY`, or `LLM_BACKEND=openai` with
`OPENAI_API_KEY`) and a search key (`LINKUP_API_KEY`; `EXA_API_KEY` and others
are optional). Then:

```bash
# Credential-free validation with dotenv loading disabled
PYTHON_DOTENV_DISABLED=1 LLM_BACKEND=gemini GEMINI_API_KEY= \
  deepsearch-run tests/fixtures/missions/test_mission.yaml --dry-run

# Run it (this makes paid model and search calls)
deepsearch-run path/to/mission.yaml
```

The credential-free command uses the legacy Gemini configuration only to validate
the local file. Production provider configurations validate their credentials
even during dry-run; a configured compiler can make model calls.

Runtime output defaults to `checkpoints/` in the working directory. Explicit
paths take precedence over per-kind overrides, which take precedence over
`DEEPSEARCH_DATA_ROOT`. Worker attempts keep their own sandbox paths. See
[docs/data_locations.md](docs/data_locations.md) to change where they go.

## Run a worker for TraceLab

In production DeepSearch runs as a long-lived worker next to a TraceLab
deployment. This command claims queued work and may incur paid calls; it is a
configuration example, not an offline documentation check:

```bash
python -m deepsearch.worker
```

TraceLab doesn't call DeepSearch over HTTP. Both services share TraceLab's
PostgreSQL database:

1. TraceLab writes an authored mission into its `missions` table and marks it
   `queued`. TraceLab vendors DeepSearch's contract compiler. Preview/compiler parity
   is a separate acceptance check; shared source alone does not prove it.
2. The worker polls for queued missions and claims one with a time-limited
   lease (120 s lease, 30 s heartbeat, three attempts).
3. Each attempt runs in a separate spawned process with its own scratch
   directory, which is removed afterwards.
4. The worker writes the report and result protocol back to the mission row
   in one transaction. Only after that commit does it send TraceLab an
   HMAC-signed completion receipt. It also streams mission logs and enrolls
   the result's evidence for TraceLab's Evidence Ledger.

At startup the worker runs 12 fail-closed preflight checks and won't claim work
until all pass. It needs:

| Setting | Purpose |
| --- | --- |
| `DATABASE_URL` | TraceLab's PostgreSQL, with TraceLab's migrations applied |
| `LLM_BACKEND` plus `DEEPSEEK_API_KEY` or `OPENAI_API_KEY` | Model provider for compiler, synthesis and critique |
| `LINKUP_API_KEY` | Primary web search |
| `TRACELAB_BASE_URL` | TraceLab API; HTTPS required outside localhost |
| `TRACELAB_API_KEY` | A TraceLab API key whose principal has the `service` role |
| `TRACELAB_DEEPSEARCH_SERVICE_SECRET` | Shared secret for signing completion receipts; must match TraceLab's configuration |

A health endpoint listens on `HEALTH_PORT` (default 8080) at `/health` and
reports preflight readiness. The included `Dockerfile` and `railway.json` run
the worker as a container; see [docs/railway-deployment.md](docs/railway-deployment.md).

## Development

```bash
python -m pytest tests -ra   # offline; ten explicitly gated live cases skip
python scripts/check_no_credential_literals.py --tracked
python -m compileall -q deepsearch
```

The offline suite supplies test-only provider configuration and needs no real
credentials. Ten opt-in live cases skip by default: nine Gemini cases and one
TraceLab PEDR case. CI runs all unit tests, two integration boundary files
(`test_s94_scope_pipeline.py` and `test_s93_acceptance.py`) and the tracked
credential scanner. It does not run the entire integration or legacy suite.
Any failed, errored or skipped unit test fails CI.

Private repair acceptance is recorded in `docs/release_verification.md`.
Public scope keeps captured vendor documentation and private research replays in
the private suite. The per-case census records every exclusion. Public creation
and pushing require a separate release approval. See [docs/public_export.md](docs/public_export.md).

Further reading:

- [docs/authored_scope.md](docs/authored_scope.md): how source and length limits are enforced
- [docs/s96_quality_and_diagnostics_contract.md](docs/s96_quality_and_diagnostics_contract.md): quality and failure telemetry consumed by TraceLab
- [docs/mission_log_delivery.md](docs/mission_log_delivery.md): live log delivery to TraceLab

## Contributing and license

The public repository is a downstream snapshot of the private working
repository; read
[CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. DeepSearch's original code uses the [MIT License](LICENSE); see [NOTICE](NOTICE) for
third-party material.
