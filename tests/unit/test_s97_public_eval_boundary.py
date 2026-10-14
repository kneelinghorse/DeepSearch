"""Omitted private evaluation tools fail with a deliberate public CLI response."""
from pathlib import Path

from deepsearch.eval import __main__ as cli


def test_omitted_archived_comparison_explains_the_public_boundary(monkeypatch, capsys) -> None:
    exists = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda path: False if path.name == "comparison.py"
                        else exists(path))
    assert cli.main(["comparison"]) == 2
    assert "only in the private repository" in capsys.readouterr().err
