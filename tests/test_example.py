"""Tests for ``clampValue`` -- written test-first (red), then implemented to green.

See ``src/myProject/example.py`` and the "Testing" section of CONVENTIONS.md.
"""

from __future__ import annotations

import myProject.example as example


def testClampValue_returnsValueWhenInRange():
    assert example.clampValue(0.5, 0.0, 1.0) == 0.5


def testClampValue_clampsBelowLow():
    assert example.clampValue(-1.0, 0.0, 1.0) == 0.0


def testClampValue_clampsAboveHigh():
    assert example.clampValue(2.0, 0.0, 1.0) == 1.0


def testClampValue_handlesEqualBounds():
    assert example.clampValue(5.0, 1.0, 1.0) == 1.0
