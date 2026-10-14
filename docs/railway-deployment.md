# Railway worker deployment

The included Dockerfile runs `python -m deepsearch.worker` on Python 3.11.
It polls TraceLab PostgreSQL and exposes `/health` on `HEALTH_PORT` (8080 by
default). Running it claims queued missions and can incur model/search charges.
The worker command is a configuration example, not an offline setup check.

## Configuration

Set `DATABASE_URL`, `TRACELAB_BASE_URL`, a TraceLab service-principal credential,
`TRACELAB_DEEPSEARCH_SERVICE_SECRET` for signed completion receipts, and
`LINKUP_API_KEY` for retrieval. Select `LLM_BACKEND=deepseek` with
`DEEPSEEK_API_KEY`, or `LLM_BACKEND=openai` with `OPENAI_API_KEY`.
See [model configuration](model_configuration.md) for supported IDs, token
budgets and precedence. Set `WORKER_LOG_FORMAT=json` for structured logs.
Use the existing TraceLab schema and lease/outbox indexes; this repository does
not create the host application's production database for you.

## Startup and verification

Startup opens health in a not-ready state and runs preflight before constructing
the polling loop. Preflight v5 checks 12 prerequisites: secure TraceLab transport,
effective model, retrieval credentials, service authentication, completion receipt,
database configuration, lease policy, execution overlap safety, database health,
mission schema, lease indexes and Evidence Ledger delivery. `/health` returns
200 when ready and 503 when not ready. Its preflight record and the redacted
`worker_preflight` log explain failures without exposing secret values.

Preflight checks provider configuration and credential structure without inference.
It does not prove account/model access or research quality. For an authorized
deploy, check readiness, exact packaged source hash, effective provider settings
and unchanged lease/fence/outbox contracts. Verify mission runtime identity and
lifecycle delivery on the next useful authorized mission.

## Operations and rollback

Deployment should wait for current work to drain. Do not assume a platform restart
will let paid work finish. Lease-v2, spawned execution, cancellation and stale-token
checks fence displaced attempts from terminal writes. Attempt files use isolated
sandboxes; see [data locations](data_locations.md).

Only `completed` and `validation_failed` artifacts enroll in the Ledger outbox;
`blocked` does not. Receipt acknowledgement and Ledger reconciliation are separate
checks. Ready health does not certify delivery for an individual mission.
Preserve paid failures and rejected artifacts. Do not restart billed research
from step one. Pin the intended source commit, provider configuration and database
contract before rollback, then verify the restored source hash and readiness.
