"""Temporary tools to validate a radar simulator."""

from __future__ import annotations

from simValidation.backProject import backProjectAll, backProjectLook, formGrid
from simValidation.matchFilter import buildFastTimeAxis, estimateGateStart_s, matchFilter
from simValidation.readPulses import readPulses

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "backProjectAll",
    "backProjectLook",
    "buildFastTimeAxis",
    "estimateGateStart_s",
    "formGrid",
    "matchFilter",
    "readPulses",
]
