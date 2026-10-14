"""Read-only Markdown citation surface scan; no claim-support inference."""

from __future__ import annotations

import re

_MARKER = re.compile(r"\[(\d+)\]|([⁰¹²³⁴⁵⁶⁷⁸⁹]+)")
_SUPER = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def reader_markers(text: str, references_range: tuple[int, int] = (0, 0)) -> list[int]:
    """Read bracket/superscript markers outside references, code and inline links.

    Numeric notation in prose can be ambiguous; this is a surface diagnostic,
    not a Markdown renderer or an inference that every number cites evidence.
    Character masking preserves offsets into the exact persisted string.
    """
    masked = list(text)

    def hide(start: int, end: int) -> None:
        masked[start:end] = ["\n" if c == "\n" else " " for c in text[start:end]]

    hide(*references_range)
    fence: tuple[str, int, int] | None = None
    offset = 0
    for line in text.splitlines(keepends=True):
        match = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence is not None:
            if match and match[1][0] == fence[0] and len(match[1]) >= fence[1]:
                hide(fence[2], offset + len(line))
                fence = None
        elif match:
            fence = (match[1][0], len(match[1]), offset)
        elif line.startswith(("    ", "\t")):
            hide(offset, offset + len(line))
        offset += len(line)
    if fence is not None:
        hide(fence[2], len(text))
    for match in re.finditer(r"(`+)(?!`)[\s\S]*?(?<!`)\1(?!`)", "".join(masked)):
        hide(*match.span())
    # Direct numeric links resolve through their own destinations, not References.
    for match in re.finditer(r"\[\d+\]\([^\n]*?\)|<!--[^>]*-->", "".join(masked)):
        hide(*match.span())
    result: list[int] = []
    visible = "".join(masked)
    for match in _MARKER.finditer(visible):
        backslashes = len(visible[:match.start()]) - len(visible[:match.start()].rstrip("\\"))
        if backslashes % 2:
            continue
        result.append(int(match[1] or match[2].translate(_SUPER)))
    return result
