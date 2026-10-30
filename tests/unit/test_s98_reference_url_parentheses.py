"""S98-m07: References URLs keep balanced parentheses and shed closing delimiters.

Balanced parentheses are legal URL characters (Wikipedia "Foo_(bar)", archive.org
"/(57-59)%20…"); a ")" that closes a Markdown link or prose aside is not part of
the URL (S84 m02 rule, applied here to the numbered-reference parser).
"""

from __future__ import annotations

import pytest

from deepsearch.agent.citation_extraction import _build_references_map


@pytest.mark.parametrize(("entry", "url"), [
    ("[1] Foo — https://en.wikipedia.org/wiki/Foo_(bar)", "https://en.wikipedia.org/wiki/Foo_(bar)"),
    ("[1] Scan — https://archive.org/stream/a/(57-59)%20B_djvu.txt",
     "https://archive.org/stream/a/(57-59)%20B_djvu.txt"),
    ("[1] [Title](https://example.com/p)", "https://example.com/p"),
    ("[1] Title (see https://example.com/q).", "https://example.com/q"),
    ("[1] https://example.com/a)", "https://example.com/a"),
    ("[1] Title — https://example.com/r, accessed 2026-10-28", "https://example.com/r"),
])
def test_reference_urls_keep_balanced_parentheses_only(entry: str, url: str) -> None:
    references, _ = _build_references_map(f"Claim [1].\n\n## References\n\n{entry}\n")
    assert references == {1: url}
