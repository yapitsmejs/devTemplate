"""Range-compress the simulator's raw LFM pulses against the reference chirp.

Reuses ``sarIfp.sarSim.applyMatchedFilter`` (frequency-domain correlation
``ifft(fft(s, axis=0) * conj(fft(h)))`` per pulse). A prior manual attempt failed
to compress because it used ``correlate(s, conj(h[::-1]))`` -- a double reversal
that convolves instead of correlating; the frequency-domain form here and in
sar-ifp is the correct one (compresses to a 2-sample peak, rel width ~0.002).

This module also owns the signal-model alignment the backprojection stage needs:
the fast-time axis is absolute seconds-since-transmit, and the receive range-gate
start is calibrated so the compressed peak maps to the known target slant range.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sarIfp import sarSim

__all__ = ["matchFilter", "buildFastTimeAxis", "estimateGateStart_s"]

# Guarded cuPy import mirroring sar-ifp: a usable device is required, not just an
# importable module. `cp` is typed Any so the cupy API is usable without mypy friction.
try:  # pragma: no cover
    import cupy as _cp

    try:  # pragma: no cover
        _HAVE_CUPY_GPU: bool = _cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # pragma: no cover
        _HAVE_CUPY_GPU = False
except ImportError:  # pragma: no cover
    _cp = None
    _HAVE_CUPY_GPU = False

cp: Any = _cp


def _matchFilterNumpy(s: np.ndarray, h: np.ndarray) -> np.ndarray:
    """CPU matched filter, identical to sar-ifp's core (the force-CPU path).

    ``ifft(fft(s, axis=0) * conj(fft(h)))`` -- the frequency-domain cross-correlation
    with ``h`` (conjugated in the frequency domain). This is the verified-correct
    form; it is NOT ``correlate(s, conj(h[::-1]))`` (that double-reverses and convolves).
    Mirrors sar-ifp: when ``s`` is 2-D, the reference spectrum is broadcast as a
    column so one 1-D ``h`` is applied per pulse (per column, axis-0 = fast-time).
    """
    spectrum = np.fft.fft(s, axis=0)
    refSpectrum = np.conj(np.fft.fft(h))
    if spectrum.ndim > 1:
        refSpectrum = refSpectrum[:, np.newaxis]
    return np.fft.ifft(spectrum * refSpectrum, axis=0)


def matchFilter(
    pulses: np.ndarray,
    referencePulse: np.ndarray,
    metadata: dict[str, Any],
    useGpu: bool | None = None,
) -> np.ndarray:
    """Range-compress the raw pulses against the LFM reference, per look.

    Args:
        pulses: Raw pulse cube ``(nLooks, pulsesPerLook, nRangeSamples)``
            complex128 (typically the mmaped ``rx_sig`` or a slice of it).
        referencePulse: The LFM reference chirp ``(nRangeSamples,)`` complex128
            (``lfm_ref``), used as the matched-filter template per pulse.
        metadata: Parsed simulator metadata. Unused by the compression itself;
            kept in the signature for symmetry with the other pipeline steps.
        useGpu: GPU selection. ``None`` auto-uses cupy when a usable GPU is
            available, else numpy. ``True`` forces the GPU path via
            ``sarIfp.sarSim.applyMatchedFilter``; ``False`` forces the CPU path.

    Returns:
        The range-compressed signal ``(nLooks, nRangeSamples, pulsesPerLook)``
        complex128 (fast-time on axis 1, slow-time/pulse on axis 2), so that
        ``compressed[look]`` is directly the ``(nRange, nPulses)`` array that
        ``sarIfp...backProjection.formImage`` consumes.
    """
    del metadata
    if useGpu is None:
        useGpu = _HAVE_CUPY_GPU
    nLooks = pulses.shape[0]
    nRangeSamples = referencePulse.shape[0]
    pulsesPerLook = pulses.shape[1]
    out = np.empty((nLooks, nRangeSamples, pulsesPerLook), dtype=np.complex128)
    for look in range(nLooks):
        # On-disk look slice is (pulsesPerLook, nRangeSamples); transpose to
        # (nRangeSamples, pulsesPerLook) so fast-time is axis-0 for the filter.
        sLook = np.asarray(pulses[look]).T
        if useGpu and _HAVE_CUPY_GPU:
            compressed = np.asarray(sarSim.applyMatchedFilter(sLook, referencePulse))
        else:
            compressed = _matchFilterNumpy(sLook, referencePulse)
        out[look] = compressed
    return out


def estimateGateStart_s(compressedLook: np.ndarray, metadata: dict[str, Any]) -> float:
    """Estimate the receive range-gate start so the peak maps to the target range.

    The compressed peak's fast-time index ``idxPeak`` is a within-gate delay; the
    target's absolute two-way delay is ``2 * slantRange_m / c``. Anchoring the gate
    start so the peak lands there yields the absolute fast-time axis that
    ``backProjection.formImage`` expects (it interpolates the compressed signal at
    the geometric delay, which must coincide with the peak).

    Args:
        compressedLook: A single look's range-compressed signal
            ``(nRangeSamples, pulsesPerLook)`` complex128.
        metadata: Parsed simulator metadata; reads ``slantRange_m``,
            ``rxSamplingFreq_Hz``, and ``wavePropogationSpeed_mPerSec``.

    Returns:
        The gate-start time in seconds (absolute, tx-referenced).
    """
    c = float(metadata["wavePropogationSpeed_mPerSec"])
    fs = float(metadata["rxSamplingFreq_Hz"])
    slantRange_m = float(metadata["slantRange_m"])
    mag = np.abs(compressedLook)
    if mag.ndim == 2:
        # Sum across pulses so the dominant fast-time peak dominates the argmax.
        mag = mag.sum(axis=1)
    idxPeak = int(np.argmax(mag))
    return 2.0 * slantRange_m / c - idxPeak / fs


def buildFastTimeAxis(metadata: dict[str, Any], gateStart_s: float) -> np.ndarray:
    """Build the absolute, tx-referenced fast-time axis (seconds).

    Matches sar-ifp's convention: ``fastTimeAxis_s = arange(N) / fs + gateStart_s``.
    With ``gateStart_s`` from :func:`estimateGateStart_s`, the compressed peak's
    index maps to the target's two-way delay, so ``range = c * t / 2``.

    Args:
        metadata: Parsed simulator metadata; reads ``nRangeSamples`` and
            ``rxSamplingFreq_Hz``.
        gateStart_s: Receive range-gate start time in seconds (absolute).

    Returns:
        ``(nRangeSamples,)`` float64 fast-time axis, seconds since transmit.
    """
    fs = float(metadata["rxSamplingFreq_Hz"])
    nRangeSamples = int(metadata["nRangeSamples"])
    return np.arange(nRangeSamples, dtype=np.float64) / fs + gateStart_s
