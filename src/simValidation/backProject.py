"""Per-look time-domain backprojection into a shared ground-plane grid.

Forms one focused image per look (361 total) by feeding each look's range-compressed
signal into ``sarIfp``'s ``formImage``. This is Phase 3 of the validation pipeline:
Phase 1 reads the pulses, Phase 2 range-compresses them, and this module focuses.

Key wiring facts (source-confirmed in sar-ifp ``backProjection.py``):
- ``formImage`` takes ``rangeCompressedSignal (nRange, nPulses)``, ``fastTimeAxis_s
  (nRange,)``, ``txPos (3, nPulses)``, ``projectionGridCoordinates (3, H, W)``, a
  ``preprocessingParams`` dict, the propagation speed, center frequency and PRF. It
  returns a focused ``(H, W)`` image. There is no ``formGrid`` in sar-ifp -- the caller
  builds the ``(3, H, W)`` grid, so :func:`formGrid` does that here.
- The per-pulse phase term is ``exp(2j*pi*fc*delay)`` only; ``pulseRepfreq_Hz`` is **not**
  used in the core loop. It only matters on the ``getBistaticRx: True`` branch, which we
  do not use (monostatic). PRF is therefore vestigial; a nominal 1.0 is passed.
- ``formImage`` interpolates the compressed signal at each pixel's geometric two-way
  delay with ``left=0, right=0`` -- a pixel whose delay falls outside the fast-time axis
  contributes **zero** (a hard range-gate cutoff, not a clamp). The grid is sized so the
  response region stays inside the gate; see :func:`formGrid`.
- ``projectionGrid`` defaults to ``None`` so each call gets a fresh zero grid; passing a
  shared grid would accumulate in place across looks.

The compressed peak index drifts across looks (~987 at look 0 to ~993 at look 360, ~1.5 m
of range walk), so :func:`backProjectAll` re-anchors the fast-time axis per look via
:func:`estimateGateStart_s` -- each look's peak self-maps to the nominal slant range and
focuses at the grid origin.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sarIfp.imageFormationProcessor.backProjection.backProjection import formImage

from simValidation.matchFilter import (
    buildFastTimeAxis,
    estimateGateStart_s,
    matchFilter,
)

__all__ = ["formGrid", "backProjectLook", "backProjectAll"]

# Monostatic matched-filter preprocessing params. ``getBistaticRx: False`` makes
# ``rxPos = txPos`` and renders ``pulseRepfreq_Hz`` vestigial (it never enters the phase
# term on this branch).
_PREPROCESSING_PARAMS: dict[str, Any] = {
    "pulseCompressionMethod": "match",
    "getBistaticRx": False,
}

# PRF is not in the data and is unused on the monostatic branch; pass a nominal value so
# ``formImage``'s signature is satisfied. Any positive float gives an identical image.
_NOMINAL_PRF_Hz: float = 1.0


def formGrid(
    metadata: dict[str, Any],
    halfExtentRange_m: float = 2.0,
    halfExtentCrossRange_m: float = 150.0,
    spacingRange_m: float = 0.1,
    spacingCrossRange_m: float = 3.0,
    azimuthDeg: float | None = None,
) -> np.ndarray:
    """Build the shared ground-plane projection grid in ENU, ``(3, nRange, nCross)``.

    The grid is a ``Z = 0`` ground plane centered on the target (origin), aligned to the
    range / cross-range directions at ``azimuthDeg`` (defaults to the collection's
    ``azimuthCenter_deg``). Range is the direction toward the platform; cross-range is
    90 deg CCW of it. Axis 0 is XYZ/ENU ``[east, north, up]``, axis 1 is range, axis 2 is
    cross-range -- the layout ``formImage`` consumes, and the layout the focused image
    comes back as.

    A square ENU grid cannot both resolve the coarse (~86-90 m) cross-range response and
    stay inside the ~3.25 m far-side range gate (range/cross-range are diagonal to ENU at
    the 45 deg look), so the grid is a range x cross-range rectangle. Cross-range offset
    is gate-invariant to first order, so it can be large; range extent is kept small.

    Args:
        metadata: Parsed simulator metadata; reads ``azimuthCenter_deg``.
        halfExtentRange_m: Half-extent of the grid in the range direction (m). The
            response main lobe is ~0.25 m; +/-2 m shows it plus near sidelobes while
            staying inside the tight far-side range gate.
        halfExtentCrossRange_m: Half-extent in the cross-range direction (m). The
            per-look cross-range resolution is ~86-90 m, so +/-150 m shows the main lobe.
        spacingRange_m: Range pixel spacing (m); 0.1 m oversamples the 0.25 m cell.
        spacingCrossRange_m: Cross-range pixel spacing (m); 3 m is well below the ~90 m
            cell.
        azimuthDeg: Look-center azimuth (deg, CCW from East). ``None`` uses
            ``metadata["azimuthCenter_deg"]``.

    Returns:
        ``(3, nRange, nCross)`` float64 ENU coordinates, ``[east, north, up]`` on axis 0.
    """
    if azimuthDeg is None:
        azimuthDeg = float(metadata["azimuthCenter_deg"])
    az = np.deg2rad(azimuthDeg)
    # Range unit vector (toward platform) and cross-range (90 deg CCW), in ENU (east,north).
    rHat = np.array([np.cos(az), np.sin(az)], dtype=np.float64)
    cHat = np.array([-np.sin(az), np.cos(az)], dtype=np.float64)
    rangeAxis = np.arange(-halfExtentRange_m, halfExtentRange_m + spacingRange_m, spacingRange_m)
    crossAxis = np.arange(
        -halfExtentCrossRange_m, halfExtentCrossRange_m + spacingCrossRange_m, spacingCrossRange_m
    )
    rangeCoords, crossCoords = np.meshgrid(rangeAxis, crossAxis, indexing="ij")
    east = rHat[0] * rangeCoords + cHat[0] * crossCoords
    north = rHat[1] * rangeCoords + cHat[1] * crossCoords
    up = np.zeros_like(east)
    return np.stack([east, north, up], axis=0)


def backProjectLook(
    compressedLook: np.ndarray,
    fastTimeAxis_s: np.ndarray,
    txPosLook: np.ndarray,
    grid: np.ndarray,
    metadata: dict[str, Any],
) -> np.ndarray:
    """Focus one look's range-compressed signal into the shared grid.

    Thin wrapper over ``sarIfp...formImage`` with the monostatic matched-filter params
    fixed. ``formImage`` auto-dispatches numpy/cupy from its module-level GPU flag and the
    input array type; passing a numpy ``compressedLook`` returns a numpy ``(H, W)`` image.

    Args:
        compressedLook: One look's range-compressed signal ``(nRange, nPulses)``
            complex128 (e.g. ``matchFilter(...)[look]``).
        fastTimeAxis_s: Absolute tx-referenced fast-time axis ``(nRange,)`` float64,
            calibrated so the compressed peak maps to the target slant range.
        txPosLook: Platform ENU position per pulse ``(3, nPulses)`` float64 (i.e.
            ``metadata["sensorPositions_ENU_m"][look].T``).
        grid: Projection grid ``(3, H, W)`` float64 ENU from :func:`formGrid`.
        metadata: Parsed simulator metadata; reads ``wavePropogationSpeed_mPerSec`` and
            ``radarCenterFrequency_Hz``.

    Returns:
        Focused image ``(H, W)`` complex128.
    """
    # ``formImage`` is untyped (sar-ifp ships no stubs) and may return a cupy array on the
    # GPU path; ``np.asarray`` coerces to a numpy ``ndarray`` (no-op for numpy input) and
    # pins complex128, satisfying the declared return type.
    return np.asarray(
        formImage(
            rangeCompressedSignal=compressedLook,
            fastTimeAxis_s=fastTimeAxis_s,
            txPos=txPosLook,
            projectionGridCoordinates=grid,
            preprocessingParams=_PREPROCESSING_PARAMS,
            wavePropogationSpeed_mPerSec=float(metadata["wavePropogationSpeed_mPerSec"]),
            radarCenterFrequency_Hz=float(metadata["radarCenterFrequency_Hz"]),
            pulseRepfreq_Hz=_NOMINAL_PRF_Hz,
        ),
        dtype=np.complex128,
    )


def backProjectAll(
    pulses: np.ndarray,
    referencePulse: np.ndarray,
    metadata: dict[str, Any],
    grid: np.ndarray | None = None,
    lookIndices: list[int] | np.ndarray | None = None,
    useGpu: bool | None = None,
) -> np.ndarray:
    """Focus every selected look into the shared grid, streaming one look at a time.

    Per look: range-compress that look only, re-anchor the fast-time axis to that look's
    compressed peak (absorbing the per-look peak-index drift), and backproject. Only one
    look's compressed signal (~19 MB) is resident at a time -- the 6.9 GB full compressed
    cube is never materialized.

    Args:
        pulses: Raw pulse cube ``(nLooks, pulsesPerLook, nRangeSamples)`` complex128
            (the mmaped ``rx_sig`` or a slice).
        referencePulse: LFM reference chirp ``(nRangeSamples,)`` complex128.
        metadata: Parsed simulator metadata; reads ``sensorPositions_ENU_m``,
            ``azimuthCenter_deg``, and the scalar signal/geometry params.
        grid: Projection grid ``(3, H, W)`` float64 from :func:`formGrid`. ``None``
            builds one with the default extents/spacing.
        lookIndices: Looks to process. ``None`` processes all looks
            ``range(metadata["nLooks"])``; pass a subset for a quick CPU check. Returns
            are stacked in this order.
        useGpu: GPU selection forwarded to :func:`matchFilter` for compression (``None``
            auto-uses cupy when available). ``formImage`` dispatches its own backend.

    Returns:
        Focused image stack ``(len(lookIndices), H, W)`` complex128.
    """
    if grid is None:
        grid = formGrid(metadata)
    nLooks = int(metadata["nLooks"])
    if lookIndices is None:
        lookIndices = np.arange(nLooks)
    lookIndicesArr = np.asarray(lookIndices, dtype=np.intp)
    sensorPositions = np.asarray(metadata["sensorPositions_ENU_m"])

    h, w = grid.shape[1], grid.shape[2]
    stack = np.empty((lookIndicesArr.shape[0], h, w), dtype=np.complex128)
    for outIdx, look in enumerate(lookIndicesArr):
        compressedLook = matchFilter(
            pulses[int(look) : int(look) + 1], referencePulse, metadata, useGpu=useGpu
        )[0]
        gateStart_s = estimateGateStart_s(compressedLook, metadata)
        fastTimeAxis_s = buildFastTimeAxis(metadata, gateStart_s)
        txPosLook = np.asarray(sensorPositions[int(look)]).T  # (3, nPulses)
        stack[outIdx] = backProjectLook(compressedLook, fastTimeAxis_s, txPosLook, grid, metadata)
    return stack
