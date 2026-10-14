"""Runtime output defaults, independent of the project record directory."""

from __future__ import annotations

import os
from pathlib import Path


def data_path(*parts: str) -> Path:
    """Resolve an output below DEEPSEARCH_DATA_ROOT at the time of use.

    Callers apply explicit paths and per-kind environment overrides first.
    Relative roots, including the default checkpoints/, follow the working
    directory. This helper does not create directories.
    """
    root = os.getenv("DEEPSEARCH_DATA_ROOT", "").strip() or "checkpoints"
    return Path(root).expanduser().joinpath(*parts)
