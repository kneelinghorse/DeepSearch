# Release verification

CI (`.github/workflows/offline-verification.yml`) runs every unit case, the
S94 authored-scope and S93 acceptance integration files, and the tracked
credential scanner. It does not run the full integration or legacy suite.
`scripts/check_unit_failure_baseline.py` rejects any failed, errored or skipped
unit test; the historical-red set is empty. The S87 CAT readiness pins were
formally retired. The private S93 comparison guard checks frozen inputs and
refuses changed code instead of asserting equality to the current tree.

## S97 local verification

The repaired private suite passed **3,774 tests**, with the same ten opt-in live
skips and twelve warnings: 3,377 unit, 237 integration and 160 legacy passes.
A fresh Python 3.11.9 environment installed `requirements.txt` and `.[dev]` in a
separate public-scope tree. That suite passed **3,656 tests**, with the same ten
skips and twelve warnings: 3,279 unit, 217 integration and 160 legacy passes.
These are local offline results, not CI, deployment or paid model observations.

The exact per-case census is `s97_test_census.json`. All 3,737 prior private cases
remain after the path split; 47 exporter/CLI-boundary cases were added. Every
relocated function's AST, including assertions and parametrization, matches the
accepted r04 tree. The public scope omits 118 explicitly identified cases with
private inputs or archived evaluation tools. No replacement production success
fixture was created. The separate private CMOS uploader suite last passed its
nine cases during r02; it is outside both published suite totals.

The ten default skips are nine cases in
`tests/integration/test_google_search_integration.py` and
`tests/integration/test_pedr_graph_integration.py::test_pedr_graph_search_live`.
Their full names and reasons are in the census. No unit case was skipped.

Documented credential-free CLI validation, authored-scope parity checking and
Python compilation were executed with no record folders. Production-provider
dry-run validates credentials, so the documented offline command deliberately
uses the legacy Gemini configuration without a key and with dotenv disabled.
A configured compiler can call a provider even during dry-run. Paid research,
worker startup, deployments and live tests were not run.

Exporter tests cover destination rejection, ignored/untracked preservation,
managed deletion, failure before and during writes, exact Git index content,
clean-filter rejection, late NUL, invalid UTF-8, unsupported selected archive
entries, executable modes, deterministic output and independent history.
Gitleaks 8.30.1 scanned the selected stage with full redaction and no unexpected
findings. The three deliberate invalid controls remain narrowly pinned by file
hash, rule and line range; redacted reports and their dispositions live outside
the source repository. The semantic tracked-tree guard remains separate.

The dependency inventory records the fresh environment, not the older developer
venv. Optional `requirements-dev.txt` pins were not installed by README/CI setup;
version differences and its missing types stub are explicit in the inventory.
No dependency trees, wheel archives, model weights or captured vendor pages are
redistributed. Their licenses are not replaced by the project's MIT grant.

The export receipt identifies the exact source commit, tree and file manifest.
Publication is a separate authorized step. The source manifest is reviewed in
the private repository; a passing reduced suite alone does not authorize a
fixture's publication or prove model behavior, compiler deployment parity,
consumer UI acceptance, or paid convergence savings.

## Production release evidence

For an authorized production release, record source/merge commits, deployment
identity, exact local and deployed build hashes, readiness/preflight, the useful
mission used for acceptance, materialization, completion receipt and Ledger
reconciliation. A successful platform deployment alone does not prove runtime
acceptance. Runtime hash verification piggybacks a useful authorized mission;
never dispatch a paid mission solely to discover its build hash.
