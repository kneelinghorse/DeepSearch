"""Scratchpad tools — closure-captured key/value store for the CodeAgent.

Verbatim behavioral port of the S62 in-runner scratchpad
(``deepsearch/agent/codeagent_runner.py`` lines 488-513). The factory builds a
fresh dict and binds two callables to it, so state lives across tool invocations
within one CodeAgent run and is wiped between runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(slots=True)
class ScratchpadBundle:
    """Bundle of (write, read, state) returned by ``build_scratchpad``."""

    write: Callable[[str, str], str]
    read: Callable[[str], str]
    state: dict[str, str] = field(default_factory=dict)


def build_scratchpad() -> ScratchpadBundle:
    """Build a fresh scratchpad and return its (write, read, state) bundle.

    The state dict is exposed for telemetry inspection by the runner; do not
    mutate it from outside the tools.
    """

    state: dict[str, str] = {}

    def scratchpad_write(key: str, value: str) -> str:
        """Persist a value under a key in the agent's scratchpad. Overwrites prior value.

        Args:
            key: Scratchpad key (e.g. "aws_lambda_notes", "matrix_draft_v1").
            value: String value to store.

        Returns:
            Confirmation string with key + value length.
        """
        if not isinstance(key, str) or not key.strip():
            raise ValueError("key must be a non-empty string")
        if not isinstance(value, str):
            raise ValueError("value must be a string")
        state[key] = value
        return f"OK: scratchpad['{key}'] = <{len(value)} chars>"

    def scratchpad_read(key: str) -> str:
        """Read a previously-written scratchpad value. Returns empty string if absent.

        Args:
            key: Scratchpad key to read.
        """
        if not isinstance(key, str):
            raise ValueError("key must be a string")
        return state.get(key, "")

    return ScratchpadBundle(write=scratchpad_write, read=scratchpad_read, state=state)


__all__ = ["ScratchpadBundle", "build_scratchpad"]
