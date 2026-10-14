# Authored research scope v1

Compiler schema **1.2**, revision **3**, carries `authored_scope` in the contract
identity. Configured entity enrichment cannot alter it. A reference URL or bound
change changes contract identity. The structural compiler manifest includes
`deepsearch/mission/scope.py`.

TraceLab rows can carry the structured object in `context.authored_scope` without
a database migration. Preview and execute requests accept `authored_scope`
(`authoredScope` alias); YAML accepts it under `context`. The typed fields are:

| Field | Meaning |
| --- | --- |
| `version` | `authored-scope-v1` |
| `reference_urls` | Exact reference seeds; never an implicit allowlist |
| `restriction` | `unrestricted`, `domains`, or `exact_pages` |
| `allowed_urls` | Required nonempty list for `exact_pages` |
| `allowed_domains` | Lowercase hostnames; permits that host and its subdomains |
| `min_words`, `max_words` | Inclusive output bounds; optional nonnegative minimum and positive maximum |
| `max_sources` | Positive maximum distinct consulted pages, including snippets |

The compiler also supports a small deterministic prose grammar in objective,
success criteria, constraints and deliverables. It recognizes numeric word ranges
(`300–500 word`, including hyphen/en dash/em dash), `at least`/`minimum [of]`
and `at most`/`no more than`/`maximum [of]` word bounds, those upper-bound phrases
for `[distinct] [primary] sources`/`source pages`/`pages`, and
`stop after`/`use only`/`consult only` `[the] [N] requested|listed|provided|referenced
[source] pages|sources|URLs`. These last phrases bind the exact reference URLs.
`Use only HOST` and `Consult only HOST` bind a concrete domain. Numbers can be
digits (including thousands separators) or one through ten.

Other prose remains prompt guidance; it is not a general natural-language scope
parser. Use the structured object for other wording. Parsed numeric bounds
intersect with structured bounds. Contradictory ranges, empty exact allowlists,
invalid URLs/fields, or exact pages conflicting with allowed/excluded domains fail
compilation before provider enrichment. Background and reference titles never
create restrictions.

Page identity uses the same Pydantic HTTP URL normalization family as the Ledger,
with fragments removed for distinct-page counting. Scheme, path case, version
paths and query strings remain distinct. `www` is never folded into the apex.
Exact-page redirects outside the explicit list are rejected before target DNS or
transport, including a move from `current` to a versioned path. This version does
not infer alias permission even from a redirect; author both destinations when
both are permitted. Domain matching is permission, not source identity merging.

Exact-page missions refuse both discovery search tools before provider calls and
use `source_fetch` directly. Domain-only missions add a positive domain clause to
web search and filter every result; explicit conflicting `site:` searches are
refused. Scholar search has no supported positive domain filter and is refused
for domain-only missions. Unrestricted search and ordinary reference seeds retain
the existing research behavior. Every admitted snippet/body consumes one page
slot; rereading the same page does not. Failed fetches and liveness probes do not
count as consultation. Requests, redirects, cache reads, actor snippets and
critique evidence share the attempt-local policy. Rejections have content-free
receipts. Exact pages not retrieved and failed fetches produce unavailable-source
warnings. No source restriction is satisfied merely by two final citations.

Durable synthesis telemetry and protocol forensic metadata carry the complete
scope object/hash, consulted identities with evidence surfaces, rejection and
unavailability receipts, contract/version/hash, authoring-context hash and actor
prompt hashes. These hashes identify the current invocation; they do not recover
historical prompts.

The PostgreSQL input fixture is byte-identical to TraceLab commit
`9caa93b8f6adbf11e4431c5616b380d478f67a37`,
`cmos/reports/sprint-61/LOG-2/proposed-mission.json`, SHA-256
`310a0cdb38196e1ae1fc8e9ec0cee7eaef3206a98d0b51a8f176ef5927c05784`.
The fixture exercises deterministic compiler/rendering parity; it does not prove
that a separate deployed consumer uses the same build or reproduce a historical
paid run. Deployment and model behavior need their own acceptance evidence.

## Persisted bounds and artifact preservation

`persisted-scope-v1` uses **`unicode-whitespace-v1`** on the exact final canonical
UTF-8 report. Words are nonempty tokens separated by this frozen whitespace set:
U+0009–000D, U+001C–0020, U+0085, U+00A0, U+1680, U+2000–200A,
U+2028–2029, U+202F, U+205F and U+3000. U+200B is not whitespace. Markdown
punctuation is not removed: a standalone `|`, `[1]`, `#` or URL is a token;
`claim[cite:URL]` is one token. Titles, metadata, tables, References, availability
warnings and the generated objective/footer all count. No HTML/Markdown stripping,
word stemming, or implicit Unicode normalization is applied by the counter.

The validator records separate pre-render and persisted word counts, character
lengths and SHA-256 of exact UTF-8 bytes. **Only the persisted count controls
inclusive authored bounds**: 299 fails a minimum of 300; 300 and 500 pass a
300–500 bound; 501 fails. The existing quality-series `report_word_count` remains
the pre-render body count for historical comparability; its matching value/hash
is explicitly labeled `authored_scope_validation.pre_render`. New consumers must
use `authored_scope_validation.persisted` for the authored bound, not the legacy
quality-series number. This does not reconcile the missing historical draft.

Validation uses all consulted identities across actor attempts and critic inputs,
including unexpected pre-critique evidence rejected by the new filter. Rejected
unseen search results and refused network actions do not count as consultation.
An escaped 22-source corpus is still a 22-source violation after filtering the
critic to two allowed pages; two final citations cannot change that verdict.

Any word/source violation produces `validation_failed` plus exact reasons, while
saving the complete report and protocol. Bounded missions also retain the complete
canonical actor output before critique, its hash and count in
`original_synthesis`. No scope repair trims output or starts another actor. The
existing same-memory generation-failure finalization remains available with the
same scope prompt; automatic new-agent critique recovery is disabled for bounded
missions. Valid bounds with unavailable sources produce a `partial` scope verdict,
reader-visible availability warning and degraded delivery. Other existing fatal
quality conditions continue to win. Scope metadata survives AgentState's declared
synthesis telemetry, protocol forensic metadata and worker execution metadata;
the final outcome includes the full count/hash/scope verdict.

## TraceLab parity package

`tests/fixtures/authored_scope_v1/manifest.json` identifies
**authored-scope-parity-v1** and hashes every input, contract, prompt and expected
output file. `canonical-input.json` is the semantic state from the real worker row
converter; `canonical-contract.json` has contract ID `8ca1ebfaa604dc7e`.
It uses canonical no-enrichment metadata, not the historical executed contract.

Run from the repository root with dependencies installed:

```bash
python scripts/check_authored_scope_parity.py
python -m pytest tests/unit/test_s94_authored_scope.py tests/unit/test_s94_scope_validation.py tests/integration/test_s94_scope_pipeline.py -o addopts='' -q
```

The checker disables socket connections and DNS. Its eight policy cases cover
both requested fetches, stored rereads at the cap, version/apex substitutions,
third-party redirects, and web/scholar discovery. The integration tests additionally
run real smolagents actors with scripted responses, real critique assembly,
mock HTTP transport, rendering, protocol serialization and capturing worker writes.
Provider/model quality is not measured. The repeated `evidence` tokens in output
fixtures intentionally test boundaries, not PostgreSQL factual synthesis.

| Fixture output | Persisted words | Consulted pages | Scope verdict | Terminal outcome |
| --- | ---: | ---: | --- | --- |
| compliant | 370 | 2 | compliant | complete |
| partial | 389 | 1 | partial | complete, degraded |
| too_long | 660 | 2 | violated | validation_failed |
| 22_sources | 370 | 22 | violated | validation_failed |

Each expected JSON binds the pre-render and final texts to exact counts and hashes.
The synthetic partial preserves its text and explicitly names missing evidence;
both violated cases preserve their complete output. TraceLab should vendor schema
1.2/compiler revision3 and this package together, run the checker against its
preview path, align `mission-authoring-contract.md`, and return parity evidence.
Local fixture acceptance is separate from deployment and TraceLab acceptance.
