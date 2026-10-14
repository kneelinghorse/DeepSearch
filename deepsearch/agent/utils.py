"""Utility helpers shared across DeepSearch agent components."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def parse_json_response(content: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    """Safely parse JSON text emitted by an LLM.

    Args:
        content: Raw string returned by the model.
        fallback: Default mapping to return when parsing fails.

    Returns:
        Parsed JSON mapping, or a copy of ``fallback`` if deserialisation fails
        or the decoded payload is not a dictionary.
    """

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse JSON: %s", exc)
        return dict(fallback)
    if not isinstance(parsed, dict):
        logger.warning("LLM response did not contain a JSON object; returning fallback")
        return dict(fallback)
    return parsed


__all__ = ["parse_json_response"]
