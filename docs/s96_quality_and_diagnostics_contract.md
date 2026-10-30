# S96 quality and diagnostics consumer contract

This document describes producer fields and consumer handling. It does not
certify a separate consumer deployment or UI. No database migration or public
citation schema change is required for these additive fields.

## Quality schema 6

Read `execution_metadata.quality_record` or the identical
`result_protocol.report_metadata.forensic.quality_record`. Existing
`delivery_quality`, `quality_gate_failures` and `coverage_gate_failures` retain
their transport and artifact-preserving semantics. Never interpret `clean` as a
factual accuracy grade: it means **no detected material issue**.

The additive `verification` object has version 1, `scope` of `sampled` or
`unassessed_or_unknown`, `report_wide_accuracy="unknown"`, and null
`report_claim_count` and `coverage_ratio`. `claim_assessment` retains the original
partial/unassessed status and assessed, supported, contradicted, unassessed and
declared-context counts. These are sampled anchors, not a denominator of every
claim. A legacy record missing these fields remains unknown; do not backfill a
supported assessment or infer comprehensive verification from a prior clean tier.

Schema 6 (S102) adds verdict provenance for a version 2 claim assessment. Each
assessment record keeps the critic's own `model_verdict` beside its validated
`verdict`. Items the critic emitted that never became records are counted by
verdict: `emitted` and `dropped_by_model_verdict`. A new verdict, `inconsistent`,
marks a claim the report states differently elsewhere; it names that statement
as `conflict_anchor`, is checked by location only and never counts as assessed.
For a version 2 assessment only, `claim_assessment` adds three counts:

- `unverified_contradicted`: claims the critic called contradicted whose quote or
  anchor failed validation, or that were dropped;
- `inconsistent`: claims the critic called inconsistent, verified or not;
- `unknown_verdicts`: unassessed or dropped claims without a readable critic
  verdict. Missing or malformed provenance counts here, never as safe.

A version 1 assessment keeps exactly its schema 5 fields. It adds
`provenance_error: true` only when the version 2 builder failed and the
version 1 record was kept instead. Older rows keep their own schema version.

| Evidence | Disclosure / operational data | Actionable result |
| --- | --- | --- |
| Supported-only sampled assessment | Scope sampled; accuracy unknown | No automatic partial warning |
| Unassessed sample anchors, version 1 assessment | Count preserved | `critique_unassessed_claims_N` warning: a version 1 record cannot show whether the claim hid a finding |
| Unassessed sample anchors, version 2, whose critic verdict is supported, unassessed or declared context | Count preserved in `claim_assessment.unassessed` | No warning: sampling coverage, not an issue |
| Contradiction the critic reported that failed validation or was dropped (version 2) | `unverified_contradicted` | `critique_unverified_contradiction_N` warning, never a degrade |
| Claim the critic marked inconsistent with another statement in the report (version 2) | `inconsistent` | `critique_internal_inconsistency_N` warning, never a degrade |
| Critic verdict that cannot be read (version 2) | `unknown_verdicts` | `critique_unknown_verdicts_N` warning |
| Version 2 builder failed, version 1 record kept | `provenance_error` | `critique_assessment_provenance_error` warning |
| No assessed anchors, absent or incomplete assessment | Unassessed/unknown scope | Assessment warning, never clean by omission |
| Contradicted anchor | Count preserved | `critique_contradicted_N`; degraded when at least 2 anchors and at least 20% of supported + contradicted (S98), otherwise a warning; artifact retained |
| Discovery errors, uncited failed fetches, cache/controller refusals, failed fetch with a retained readable body | Existing `retrieval_failures` and raw tool records remain inspectable | No independent warning when final dependencies are satisfied |
| Cited URL missing from collected sources, cited source without a readable fetched body, truncated cited body, failed cited fetch still lacking a body | `evidence_dependencies` version 1 counts | Respective `missing_cited_sources_N`, `cited_without_readable_body_N`, `truncated_cited_bodies_N`, `unresolved_cited_fetch_failures_N` warnings |
| Missing/truncated failure-attribution records | Attribution explicitly incomplete | `retrieval_dependency_attribution_unknown`; older aggregates with errors use `retrieval_failure_impact_unknown` |
| Unused bibliography entries without mapping defects | `citation_surface.accounting_only=true`, surplus counters retained | No automatic mapping warning |
| Collisions, dangling numbers, unmapped citations, coordinate drift, missing quarantine/dead flags, audit failure | `citation_surface.issues` | Mapping warning |
| Hallucinated/quarantined links, unsafe URL/source projection drops, unmappable coordinates | Existing explicit counts | Counted warnings; report retained |
| Material coverage gaps, truncation, unproven recovery | Existing telemetry and severity | Existing degraded/validation-failed semantics; no new blocking gate |

All detailed warnings remain visible when coverage or another finding has already
made a report degraded. The old early return hid these independent facts.

Since S103 a superscript digit run is a reader marker only in a report that cites
that way: one that labels a References entry with a superscript, or whose only
citations are superscripts. In a report that labels its entries `[N]`, `χ²`, `10⁶`
and `p<10⁻³³` are maths, never citations or dangling numbers, and in every report a
run after `⁺` or `⁻` is an exponent. `citation_consistency.superscripts_read`
records the reading and `superscript_runs_ignored` counts the runs read as maths,
masked as the reader-marker scan masks. Both are diagnostics, never issues.

Since S103 the critique can show a cited source more of its text. When the source
packer drops no row and limits no row below the per-source cap, a cited source it
cut shows up to one 600-character chunk per citing claim, 16 at most, inside the
budget the packer left unused. Its baseline chunks come first, each widened to its
whole stride. Every other row is unchanged, and visibility and completeness stay
the baseline's, so this never authorizes an `unsupported` deletion. These
`critique_telemetry.grounding` fields describe it:

- `cited_extra_unsqueezed`: whether the baseline qualified;
- `cited_extra_remainder`: the unused budget, in characters;
- `cited_extra_eligible`: cut cited sources;
- `cited_extra_baseline_passages`: their baseline passages;
- `cited_extra_sources`: sources extended;
- `cited_extra_chunks`: extra chunks added, not passages, since touching ranges merge;
- `cited_extra_chars`: characters added to the source block;
- `cited_extra_skipped`: `remainder_below_widening` when the remainder cannot pay
  for widening, else null;
- `cited_extra_fallback`: true when the extension failed and the baseline was kept.

Each evidence-manifest passage of an extended source carries `baseline_ranges`, the
baseline passages it contains, empty for a passage of extra chunks only. Assessment
evidence entries copy their manifest entry, so they carry it too. No other passage
has the key.

Source dependency counts use the pipeline's exact canonical URLs. A readable body
requires `body_source=source_fetch`, `content_quality=readable_body`, positive body
length and no explicit gone flag. Snippets are useful discovery material but do
not count as fetched bodies. A retained readable body resolves the availability
dependency; it does not prove support for every claim citing that URL. Source
truncation still warns. No www/apex equivalence is inferred. Since S98, collection
membership alone (`missing_cited_sources` and liveness orphans) treats the arXiv
abs, pdf, html and `doi.org/10.48550/arXiv.<id>` URLs of one paper as one source;
readable, truncated and failed-fetch counts stay exact.

Generic tool records retain `network_requested`; Ledger records intentionally do
not. `source_fetch` errors with `network_requested=false` count as local refusals,
not network failures. Attribution is complete only when the generic records list
exists and `records_truncated=0`. Discovery counts, all fetched bodies, readable
cited bodies, linked citation URLs, placements and assessed anchors are different
measures and must be labeled separately.

## Provider diagnostics

Under `execution_metadata.synthesis_telemetry.provider_attempts[]`, failed events
add `failure.rejection` and `failure.request_shape`, each version 1. The same
attempt channel also carries critique and recovery attempts through existing
accounting. The envelope survives a blocked run and attempt-sandbox cleanup.

Rejection subtypes are `context_limit`, `unsupported_parameter`,
`unsupported_value`, `content_rejection`, `invalid_json`,
`invalid_message_structure`, and `unknown`. `subtype_source` is `code`, `type`,
`message_pattern`, or `unknown`. Pattern classification describes provider wording,
not an independently established root cause. `parameter` and `provider_type` are
fixed allowlisted labels; message indexes are removed. Unknown values remain
unknown. HTTP status and retryability remain available in the existing envelope.

Request shape contains availability, bounded message/role counts, text-character
count, nontext/unknown-message count, tool count and a clipping flag. It is not
token usage or wire size. Clipped counts are lower bounds; unavailable shape is
not zero. No raw request, provider message, fetched body, header, ID or fingerprint
is exposed. Compiler observations may have unavailable shape.

`generation_recovery_result=provider_failed` means an observed second provider
failure occurred during the one retained-memory finalization. `unknown_finish`
means a response returned without a proven finish reason. Neither authorizes a
fresh research run. Missing usage is still an accounting gap, not zero cost.
Historical errors without this envelope remain unknown; classification cannot
reconstruct provider explanations that were not retained.

## Consumer guidance

Consumers should:


1. Display “No detected material issue” for clean, with a separate “Sampled
   verification; report-wide accuracy unknown” disclosure. Display unknown scope
   for missing legacy data. Never show a percentage using source or sample counts.
2. Display actionable quality/coverage reasons separately from expandable raw
   operational retrieval counts and bibliography accounting.
3. On failed runs, display the fixed rejection subtype/provenance and request-shape
   availability from provider attempts. Distinguish failed finalization from an
   unknown response finish. Keep existing `error_message` sanitization unchanged.
4. Accept unknown future reason strings without dropping the report or hiding the
   raw safe telemetry. Verify all warning, degraded, legacy and blocked examples.

`tests/fixtures/s96_consumer_contract.json` provides synthetic metadata projections
for a supported sample, missing cited body, legacy uncertainty and a failed
retained-memory request. They are review fixtures, not production records.

Consumer rendering and live UI acceptance require separate verification.
