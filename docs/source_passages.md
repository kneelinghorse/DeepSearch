# Reading complete collected sources

`source_fetch(url)` performs the existing safe network fetch once per exact URL
and attempt. It stores the readable body and returns at most 12,000 characters.
Repeated default calls reuse that snapshot and its original `fetched_at`.

After fetching:

```python
found = source_fetch(url, query="incomplete", length=1000)
passage = source_fetch(url, start=found["matches"][0]["start"], length=4000)
```

Find is literal and case-insensitive, at most 200 query characters and five match
ranges. `next_match_start` continues a bounded find. Range reads use zero-based
Unicode character offsets, with an exclusive end. `start == text_chars_total`
is a valid empty final page; a larger or negative offset is an explicit error.
Find and nonzero-offset reads without a stored body return `missing_stored_body`
and perform no request. A read at offset zero retains first-fetch behavior.

Responses expose `source_id` (SHA-256 of the exact fetched final URL),
`body_sha256`, `range_start`, `range_end`, `remaining_chars`, `next_start`,
`matches_truncated`, `text_chars_total`, and `text_chars_returned`. `body_complete`
means the stored body was not truncated at the upstream extraction boundary;
`text_truncated_for_tool` describes the returned window. `search_complete`
describes a search within the stored body after the requested start only.
A match or no-match says nothing about claim support or unseen remote text.
The serialized response remains within the existing 16,000-character budget,
with the final range recomputed after any encoding-related clipping.

Use `refresh=True` alone for a new network version. A successful fetch replaces
the current stored version even when shorter. The source ID stays stable while
the body hash changes. A failed refresh preserves the prior cached version and
its old timestamp. The cache lives only within the current agent attempt.

Only successful fetch redirects establish request/final-URL aliases. No www,
apex, HTTP/HTTPS, or unobserved canonical equivalence is inferred. Existing URL
safety, excluded-domain handling, authority assignment, timeout, and cancellation
boundaries apply. Local read logs carry content-free source/version/range
provenance, but do not manufacture network attempts or LEDGER-2 HTTP outcomes.
Diagnostic sampling retains its explicit truncation counter.
