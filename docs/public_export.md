# Public snapshot and maintenance

DeepSearch's public repository is a reviewed downstream snapshot with fresh
history. The private repository remains the working and deployment source.
Public creation, pushing and visibility changes require a separate release
approval after the exact source commit and destination are known.

## Scope and evidence

The operator approved keeping real private-evidence regressions and three full
vendor-document replays private. The other 27 existing source-passage/critique
cases need no captured documentation and remain public. The public fixture review
approves 50 first-party files and withholds 110 retained files, among them every
real paid-run fixture added since S97. It does not grant redistribution rights to
vendor captures or replace private inputs with invented successful reports.

`docs/s97_test_census.json` maps all 4,615 collected private cases to their public
or private disposition and maps relocated cases to their previous identities.
The public scope has 4,085 cases. The 530 withheld cases comprise 481 retained
private-evidence cases, three vendor replays, two private-registry guards and
44 archived evaluation cases. All remain in the private suite. Relocated test
assertions and parametrization were checked against the accepted private tree.
Cases added since S97 follow their file: a test file named `*_private*` is
withheld, and every other new test reads no private fixture.

Dependencies are installed separately. `docs/dependency_licenses.json` records
versions, publisher declarations and available license-file hashes from the
fresh setup environment. It is not a lockfile or a grant for bundled dependency
redistribution. This snapshot includes no installed packages or model weights.

## Exact-file export

The private manifest names exact files, hashes, executable modes, publication
dispositions and review/license bases. The exporter reads that manifest and all
bytes from the same immutable commit. New or changed files require review. The
private fixture registry also rejects exact unapproved bytes relocated under a
new name; content review remains necessary for transformed material.

Before writing, the exporter rejects unsafe paths and selected nonregular archive
entries, checks strict UTF-8 and every byte for NUL, and scans the staged snapshot
with gitleaks. Reports are fully redacted and kept outside the repository. Three
intentional invalid credential controls are accepted only at exact file hashes,
rules and line ranges recorded in `scripts/export_public.py`. A new finding or
any changed control file fails closed. Whole directories are never exempted.

Only an empty new directory or an explicitly managed clean Git checkout is
accepted. Managed sync requires the full expected HEAD. Root, home, source
ancestors/descendants, symlink aliases, other repositories, untracked/ignored
files and unknown directories are refused. The destination is checked again
after scanning. Deletions affect only previously managed tracked paths.

After writing, indexed paths, bytes and modes must match the scanned files.
The commit uses that verified immutable tree, so Git filters and unrelated index
additions cannot change its contents. The receipt records the full source commit,
source tree and exact reviewed manifest. Initial history has no private ancestors.
The tool makes no network calls. Interrupted writes remain dirty and recoverable;
it never resets the checkout or removes unknown data during recovery.

## Maintainer commands

Run these in the private working repository with gitleaks installed. They are
parameterized release examples; replace the commit and destination deliberately:

```bash
python scripts/export_public.py --rev SOURCE_COMMIT --out /absolute/new-directory --check
python scripts/export_public.py --rev SOURCE_COMMIT --out /absolute/new-directory
python scripts/export_public.py --rev SOURCE_COMMIT --out /absolute/managed-directory --expected-head DOWNSTREAM_COMMIT
```

The manifest and private registry are withheld from the public source tree;
the public receipt contains its exact approved selection. The archived private
comparison CLI is unavailable in public builds. Exporter boundary tests create
their own disposable repositories and run without private evidence:

```bash
python -m pytest tests/unit/test_s97_public_export.py -ra
```

See `docs/release_verification.md` for execution evidence and limits. Contributor
port-back, attribution and maintainer sync ownership are in `CONTRIBUTING.md`.
