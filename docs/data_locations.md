# Runtime data locations

`DEEPSEARCH_DATA_ROOT` selects the default root for the outputs listed below.
When unset or blank it is `checkpoints/`, which is gitignored. Relative roots
follow the process working directory; `~` in the root expands to the user's
home directory. Paths are resolved when used, not at module import.

Precedence: an explicit function or CLI path wins, then an existing per-kind
environment variable, then the data-root default. Setting a root does not move
existing artifacts. Use explicit paths to read older archives.

## Writer inventory

The before column is private main `23e25906e4f25b2d19728e225d6edf35eda1a7c2`.
`<root>` means `DEEPSEARCH_DATA_ROOT` or `checkpoints/`.

| Output and writer | Before | After | Higher-priority path |
| --- | --- | --- | --- |
| Reports and protocols — `report/generator.py` | `cmos/research/` | `<root>/reports/` | `DEEPSEARCH_REPORT_DIR` |
| Extracted protocols and Markdown — `eval/protocol_extract.py` | `cmos/research/` | `<root>/protocols/` | `--out` / `out_path` |
| Hybrid validation metrics — `validation/hybrid_evaluator.py` | `cmos/research/validation/` | `<root>/validation/` | `--validation-output` / `output_dir` |
| Contract comparison JSON and Markdown — `validation/harness.py` | `cmos/research/validation/` | `<root>/validation/` | `--output` / `output_path` |
| Failed CLI uploads — `runner.py::_save_failed_submission` | `cmos/research/failed_submissions/` | `<root>/failed_submissions/` | None |
| CLI upload telemetry — `tracelab/uploader.py` | `cmos/telemetry/events/tracelab-uploads.jsonl` | `<root>/telemetry/tracelab-uploads.jsonl` | `telemetry_path`, then `DEEPSEARCH_TELEMETRY_DIR` |
| API webhook dead letters — `api/webhook.py` | `cmos/webhooks/failed/` | `<root>/webhooks/failed/` | `failed_payloads_dir` |
| Compiled contracts — `mission/contract.py` | `checkpoints/contracts/` | `<root>/contracts/` | `output_dir`, then `DEEPSEARCH_CONTRACT_DIR` |
| Mission/evidence telemetry — `agent/evidence.py` | `checkpoints/telemetry/` | `<root>/telemetry/` | `output_dir`, then `DEEPSEARCH_TELEMETRY_DIR` |
| LangGraph checkpoints — `config.py` | `checkpoints/deepsearch.db` | `<root>/deepsearch.db` | Explicit settings path, then `CHECKPOINT_DB_PATH` |

The telemetry writer previously preferred its environment override to its
explicit argument. S97-r01 corrects that ordering to match the approved
precedence above. Upload telemetry now also honors `DEEPSEARCH_TELEMETRY_DIR`.
An explicit comparison output does not create an unused default directory.

FActScore protocol discovery uses explicit `search_dirs` when supplied.
Otherwise it searches `DEEPSEARCH_REPORT_DIR` first if set, then
`<root>/protocols/`, then `<root>/reports/`. `--protocol` bypasses discovery.
The validation metrics finder uses an explicit directory or `<root>/validation/`.

This is not a blanket relocation of all application files. CLI logs,
FActScore/benchmark scoresheets, tool caches, and the worker attempt root keep
their existing paths and overrides. The calibration quality series is offline
project evidence, not a runtime writer; its path belongs to its consuming test,
not the production package.

## Worker isolation

`worker/execution.py::_configure_attempt_environment` continues to set report,
telemetry, contract and checkpoint paths inside the attempt sandbox before
mission execution. Those per-kind paths win over an inherited data root.
`WORKER_ATTEMPT_ROOT` and attempt cleanup are unchanged. Durable worker results
still go to TraceLab Postgres. CLI uploads, evaluation commands, and API webhook
writers have separate entry points; the root change does not add them to the
production worker pipeline.

## Removed record loader and verification

The private-main caller/export audit found `load_mission_from_cmos` only in its
definition, the mission package exports, and three tests of that loader.
No production caller used it. `cmos_db_path` only supplied runner result
metadata; `QUALITY_SERIES_RELPATH` only supplied an offline calibration test.
The loader, its private helpers, settings field, runner metadata keys, and unused
exports are removed. YAML and worker TraceLab mission loading remain supported.

`tests/unit/test_s97_record_folder_guard.py` checks executable record-folder
literals, imports, removed settings consumers, and the loader export surface.
`tests/unit/test_s97_data_locations.py` checks independent path precedence.
The private suite additionally preserves retained-report routing, protocol
serialization and actual worker sandbox regressions; those cases are listed in
the S97 case census and excluded from the public snapshot with their evidence.
These offline checks do not claim a deployed or paid run.
