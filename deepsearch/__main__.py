"""Module entry point to support ``python -m deepsearch``."""

from __future__ import annotations

from .cli import console_main

if __name__ == "__main__":  # pragma: no cover - dispatch only
    console_main()
