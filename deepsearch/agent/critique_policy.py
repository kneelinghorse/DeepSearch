"""Shared materiality policy for structured critique ratios."""

from __future__ import annotations

# Ratio evidence is too denominator-sensitive below three applied annotations
# to drive recovery or a catastrophic terminal outcome by itself.
CRITIQUE_RATIO_MIN_APPLIED = 3

__all__ = ["CRITIQUE_RATIO_MIN_APPLIED"]
