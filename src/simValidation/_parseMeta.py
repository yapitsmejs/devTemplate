"""Parse the simulator's text metadata sidecars into a typed scalar dict.

The simulator writes two free-form text sidecars alongside the repackaged
``.npy`` data: ``Meta_parameters.txt`` (waveform / ADC parameters) and
``geometry_note.txt`` (collection geometry). This module turns them into a
``dict[str, Any]`` whose keys follow the project's camelCase + ``_aspect``
naming (unit / coordinate-frame suffix), so downstream phases can pull values
by name without re-parsing text.

Parsing is deliberately tolerant: labels are matched case-insensitively by
prefix, units are converted to SI (Hz, s, m), and the simulator's known typo
``Rx polariation`` is handled with a fallback. Unknown units default to a
multiplier of 1.0 (so ``deg`` is returned unchanged).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["parseMeta", "parseMetaParameters", "parseGeometryNote"]

# Speed of light, m/s. Stored as a literal so the parser stays dependency-light
# and the value is explicit (matches the physical constant; sar-ifp defaults to
# 3e8 elsewhere -- reconcile in the Phase 2 fast-time spike if it matters).
_SPEED_OF_LIGHT_MPerSec: float = 299_792_458.0

# Unit token -> multiplier to the SI base (Hz, s, m). ``deg`` is intentionally
# absent so it falls through to 1.0 and is returned as-is.
_UNIT_MULTIPLIERS: dict[str, float] = {
    "GHz": 1e9,
    "MHz": 1e6,
    "kHz": 1e3,
    "Hz": 1.0,
    "μs": 1e-6,
    "µs": 1e-6,  # micro sign (U+00B5) variant
    "us": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
    "km": 1e3,
    "mm": 1e-3,
    "m": 1.0,
    "Gbps": 1e9,
    "Mbps": 1e6,
    "bps": 1.0,
}

_NUMBER_RE = re.compile(r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")
_ALL_NUMBERS_RE = re.compile(r"-?\d+\.?\d*")


def _splitLabelled(text: str) -> dict[str, str]:
    """Map a lowercase label to its raw value string for each ``label: value`` line."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        if ":" not in raw:
            continue
        label, value = raw.split(":", 1)
        out[label.strip().lower()] = value.strip()
    return out


def _field(fields: dict[str, str], key: str, source: str) -> str:
    """Return the value for ``key``, raising a clear error if it is missing."""
    try:
        return fields[key]
    except KeyError:
        raise ValueError(f"missing '{key}' field in {source}") from None


def _scalar(valueStr: str) -> float:
    """Extract the first number in ``valueStr`` and convert its unit to SI."""
    match = _NUMBER_RE.search(valueStr)
    if match is None:
        raise ValueError(f"no numeric value found in {valueStr!r}")
    value = float(match.group(1))
    rest = valueStr[match.end() :].strip()
    unit = rest.split()[0] if rest else ""
    return value * _UNIT_MULTIPLIERS.get(unit, 1.0)


def _parseFloats(valueStr: str) -> list[float]:
    """Extract every number in ``valueStr`` (e.g. ``[0, 0, 0] m`` -> [0, 0, 0])."""
    return [float(token) for token in _ALL_NUMBERS_RE.findall(valueStr)]


def parseMetaParameters(path: Path) -> dict[str, Any]:
    """Parse ``Meta_parameters.txt`` into waveform / ADC scalar parameters."""
    fields = _splitLabelled(path.read_text(encoding="utf-8"))
    source = str(path)
    rxPol = fields.get("rx polariation") or fields.get("rx polarization") or ""
    txPol = fields.get("tx polarization") or ""
    return {
        "radarCenterFrequency_Hz": _scalar(_field(fields, "center frequency", source)),
        "radarBandwidth_Hz": _scalar(_field(fields, "bandwidth", source)),
        "rxSamplingFreq_Hz": _scalar(_field(fields, "adc sampling rate", source)),
        "chirpDuration_s": _scalar(_field(fields, "chirp duration", source)),
        "pulseDuration_s": _scalar(_field(fields, "total pulse duration", source)),
        "chirpRate_HzPerSec": _scalar(_field(fields, "chirp rate", source)),
        "adcFrequencySamples": int(_scalar(_field(fields, "adc frequency samples", source))),
        "adcSamplesPerPulse": int(_scalar(_field(fields, "adc samples per pulse", source))),
        "originalNumFreqs": int(_scalar(_field(fields, "original num_freqs", source))),
        "rxPolarization": rxPol,
        "txPolarization": txPol,
    }


def parseGeometryNote(path: Path) -> dict[str, Any]:
    """Parse ``geometry_note.txt`` into collection geometry parameters."""
    fields = _splitLabelled(path.read_text(encoding="utf-8"))
    source = str(path)
    grazingAngle_deg = _scalar(_field(fields, "grazing angle", source))
    slantRange_m = _scalar(_field(fields, "slant range", source))
    groundRange_m = _scalar(_field(fields, "ground range", source))
    radarAltitude_m = _scalar(_field(fields, "radar altitude", source))
    azimuthCenter_deg = _scalar(_field(fields, "azimuth center", source))
    azimuthHalfSpan_deg = _scalar(_field(fields, "azimuth half-span", source))
    lookSpacing_deg = _scalar(_field(fields, "look spacing", source))
    pulsesPerLook = int(_scalar(_field(fields, "pulses per look", source)))
    totalPulses = int(_scalar(_field(fields, "total pulses", source)))
    pulseToPulseAzStep_deg = _scalar(_field(fields, "pulse-to-pulse az step", source))
    targetOrigin_XYZ_m = np.asarray(
        _parseFloats(_field(fields, "target origin", source)), dtype=np.float64
    )
    coordinateConvention = fields.get("coordinate convention", "")
    nLooks = totalPulses // pulsesPerLook
    return {
        "targetOrigin_XYZ_m": targetOrigin_XYZ_m,
        "radarAltitude_m": radarAltitude_m,
        "grazingAngle_deg": grazingAngle_deg,
        "grazingAngle_rad": np.deg2rad(grazingAngle_deg),
        "slantRange_m": slantRange_m,
        "groundRange_m": groundRange_m,
        "azimuthCenter_deg": azimuthCenter_deg,
        "azimuthHalfSpan_deg": azimuthHalfSpan_deg,
        "lookSpacing_deg": lookSpacing_deg,
        "pulseToPulseAzStep_deg": pulseToPulseAzStep_deg,
        "pulsesPerLook": pulsesPerLook,
        "totalPulses": totalPulses,
        "nLooks": nLooks,
        "coordinateConvention": coordinateConvention,
    }


def parseMeta(metaParamsPath: Path, geometryNotePath: Path) -> dict[str, Any]:
    """Parse both sidecars and add derived parameters shared across phases.

    Args:
        metaParamsPath: Path to ``Meta_parameters.txt``.
        geometryNotePath: Path to ``geometry_note.txt``.

    Returns:
        A merged ``dict[str, Any]`` of scalar parameters. Per-pulse geometry
        arrays are added by ``readPulses``, not here.
    """
    meta: dict[str, Any] = {}
    meta.update(parseMetaParameters(metaParamsPath))
    meta.update(parseGeometryNote(geometryNotePath))

    centerFreq_Hz = float(meta["radarCenterFrequency_Hz"])
    meta["wavePropogationSpeed_mPerSec"] = _SPEED_OF_LIGHT_MPerSec  # note: sar-ifp spelling
    meta["radarWavelength_m"] = _SPEED_OF_LIGHT_MPerSec / centerFreq_Hz
    meta["nRangeSamples"] = int(meta["adcSamplesPerPulse"])
    return meta
