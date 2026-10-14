# S96 quality and diagnostics consumer contract

This document describes producer fields and consumer handling. It does not
certify a separate consumer deployment or UI. No database migration or public
citation schema change is required for these additive fields.

## Quality schema 5

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

| Evidence | Disclosure / operational data | Actionable result |
| --- | --- | --- |
| Supported-only sampled assessment | Scope sampled; accuracy unknown | No automatic partial warning |
| Unassessed sample anchors | Count preserved | `critique_unassessed_claims_N` warning |
| No assessed anchors, absent or incomplete assessment | Unassessed/unknown scope | Assessment warning, never clean by omission |
| Contradicted anchor | Count preserved | `critique_contradicted_N`, degraded; artifact retained |
| Discovery errors, uncited failed fetches, cache/controller refusals, failed fetch with a retained readable body | Existing `retrieval_failures` and raw tool records remain inspectable | No independent warning when final dependencies are satisfied |
| Cited URL missing from collected sources, cited source without a readable fetched body, truncated cited body, failed cited fetch still lacking a body | `evidence_dependencies` version 1 counts | Respective `missing_cited_sources_N`, `cited_without_readable_body_N`, `truncated_cited_bodies_N`, `unresolved_cited_fetch_failures_N` warnings |
| Missing/truncated failure-attribution records | Attribution explicitly incomplete | `retrieval_dependency_attribution_unknown`; older aggregates with errors use `retrieval_failure_impact_unknown` |
| Unused bibliography entries without mapping defects | `citation_surface.accounting_only=true`, surplus counters retained | No automatic mapping warning |
| Collisions, dangling numbers, unmapped citations, coordinate drift, missing quarantine/dead flags, audit failure | `citation_surface.issues` | Mapping warning |
| Hallucinated/quarantined links, unsafe URL/source projection drops, unmappable coordinates | Existing explicit counts | Counted warnings; report retained |
| Material coverage gaps, truncation, unproven recovery | Existing telemetry and severity | Existing degraded/validation-failed semantics; no new blocking gate |

All detailed warnings remain visible when coverage or another finding has already
made a report degraded. The old early return hid these independent facts.

Source dependency counts use the pipeline's exact canonical URLs. A readable body
requires `body_source=source_fetch`, `content_quality=readable_body`, positive body
length and no explicit gone flag. Snippets are useful discovery material but do
not count as fetched bodies. A retained readable body resolves the availability
dependency; it does not prove support for every claim citing that URL. Source
truncation still warns. No www/apex equivalence is inferred.

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
