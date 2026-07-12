"""Example module showing the project conventions and TDD workflow.

This pair (``example.py`` + ``tests/test_example.py``) is illustrative: the test
was written first and run red, then this implementation made it green. Delete
both files once the first real module exists (see README "Starting a new
project from this template").
"""

from __future__ import annotations

__all__ = ["clampValue"]


def clampValue(value: float, low: float, high: float) -> float:
    """Clamp ``value`` to the inclusive ``[low, high]`` range.

    Args:
        value: The number to clamp.
        low: The inclusive lower bound.
        high: The inclusive upper bound.

    Returns:
        ``value`` if it lies within ``[low, high]``; otherwise the nearest
        bound.
    """
    if value < low:
        return low
    if value > high:
        return high
    return value
