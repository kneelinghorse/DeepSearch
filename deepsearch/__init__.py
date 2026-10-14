"""DeepSearch research orchestration package."""

from importlib import metadata

__all__ = ["__version__"]


def _detect_version() -> str:
    """Return the installed package version or a development fallback."""
    try:
        return metadata.version("deepsearch")
    except metadata.PackageNotFoundError:
        return "0.1.0-dev"


__version__ = _detect_version()
