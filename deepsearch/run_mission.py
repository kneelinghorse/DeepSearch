"""Legacy CLI entry point retained for backwards compatibility."""

from __future__ import annotations

from typing import Sequence

from deepsearch.runner import execute_research_mission


def main(argv: Sequence[str] | None = None) -> int:
    """Invoke the CLI runner (``python -m deepsearch.run_mission``)."""

    from deepsearch.cli import main as cli_main

    return cli_main(argv)


def cli() -> None:
    """Console script used by ``python -m deepsearch.run_mission``."""

    raise SystemExit(main())


__all__ = ["execute_research_mission", "main"]


if __name__ == "__main__":
    cli()
