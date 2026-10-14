# Contributing to DeepSearch

Thanks for your interest. Please read how this repository works before you open
a pull request, because it is not developed here directly.

## How this repository is maintained

- **The public repository is a downstream snapshot.** DeepSearch is developed in a
  private working repository, which is also the source of every production
  deployment. Each release here is one snapshot commit made by an export
  script from a specific commit of the working repository. The commit message
  names that source commit.
- **History here starts fresh.** It doesn't contain the working repository's
  history, internal project records or production evidence, and it never will.
- **Changes flow one way: working repository → this repository.** Nothing is
  pushed from here back into production.

## How contributions are accepted

1. Open an issue or a pull request here as usual. CI runs every unit test, the authored-scope and S93 acceptance integration
   boundary files, and the tracked credential scanner. Maintainers also run the
   complete private suite before release.
2. If a change is accepted, a maintainer **ports it back** into the working
   repository: the change is re-applied there, reviewed, and tested against
   the same suite.
3. The change appears here in the **next release sync**, as part of a snapshot
   commit. Your pull request is then closed with a link to that commit, rather
   than merged.

We record credit for ported contributions in the release commit message with a
`Co-authored-by:` trailer using the name and email from your commits. If you
prefer different attribution, say so in the pull request.

## What makes a change easy to port

- Keep it focused: one behavior change per pull request.
- Include tests that explain why the behavior matters. Unit tests must have
  zero failures, errors or skips. Integration has ten explicit opt-in live skips
  (nine Gemini and one TraceLab PEDR); do not hide additional skips.
- Don't add hard-coded paths into project-record folders; runtime output goes
  under `checkpoints/` (see `docs/data_locations.md`).
- Never include credentials or private reports in public fixtures. A real
  fixture needs recorded provenance and an explicitly reviewed redistribution
  basis. Do not replace production regression evidence with fabricated success
  examples to make an export pass.

## Security

Please don't report vulnerabilities in public issues. Use the repository's private security-reporting channel when it is available; do not post secrets
or vulnerability details in a public issue.

## Release sync ownership

The private release maintainer owns each downstream sync, including the reviewed
file manifest, dependency/license inventory, privacy review and complete test
census. Contributors do not merge public pull requests into release history.
The maintainer records the full private source commit/tree and exact exported
manifest in `.deepsearch-export.json`; public history starts independently.
Publication, repository creation and pushing require a separate authorized release.
Private research replays and captured vendor documentation stay in the private
suite. The public case census identifies the coverage difference; it is not a
claim that those inputs were replaced or independently revalidated.
