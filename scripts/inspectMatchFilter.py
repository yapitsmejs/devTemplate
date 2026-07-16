"""Phase 2 validation: confirm the matched filter compresses the simulator pulses.

Run under a debugger to inspect intermediate arrays, or headless to print the
acceptance report::

    uv run python scripts/inspectMatchFilter.py

Throwaway inspection script (not part of the gate). Verifies the six Phase 2
acceptance criteria from ``plans/phase2Plan.md``: the matched filter compresses
to a narrow peak, the prior manual-failure root cause is reproduced and fixed,
numpy and cupy agree, the peak is stable across pulses, the peak maps to the
known target slant range, and ``lfm_ref`` is a valid clean transmit replica.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import simValidation as sv
from simValidation.matchFilter import _HAVE_CUPY_GPU

DATA_DIR = Path(
    r"C:\Users\yJoonSio\OneDrive - DSO\trihedral"
    r"\Trihedrals_SAR_ADC\raw_pulse_data_repackaged"
)

# Pulses to spot-check for peak stability across the aperture.
_STABILITY_PULSES = (0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1199)


def _threeDbWidth(profile: np.ndarray) -> tuple[int, int, float]:
    """Return (peakIdx, 3dB-width-samples, relative-width) of a 1-D magnitude profile."""
    mag = np.abs(profile).astype(np.float64)
    idx = int(np.argmax(mag))
    half = mag[idx] / np.sqrt(2.0)
    left = idx
    while left > 0 and mag[left - 1] >= half:
        left -= 1
    right = idx
    while right < mag.shape[0] - 1 and mag[right + 1] >= half:
        right += 1
    width = right - left + 1
    return idx, width, width / mag.shape[0]


def _idealUpChirp(metadata: dict[str, Any]) -> np.ndarray:
    """Baseband up-chirp ``exp(j*pi*K*t^2)`` from metadata (K = chirp rate, t = arange/fs)."""
    fs = float(metadata["rxSamplingFreq_Hz"])
    nRangeSamples = int(metadata["nRangeSamples"])
    chirpRate_HzPerSec = float(metadata["chirpRate_HzPerSec"])
    t = np.arange(nRangeSamples, dtype=np.float64) / fs
    return np.exp(1j * np.pi * chirpRate_HzPerSec * t**2)


def main() -> int:
    pulses, referencePulse, metadata = sv.readPulses(DATA_DIR)
    h = np.asarray(referencePulse)
    sPulse0 = np.asarray(pulses[0, 0, :])  # look 0, pulse 0: (nRange,)

    results: list[tuple[str, bool, str]] = []

    # --- 1. Compression works (numpy, full look 0) ---
    compressedLook = sv.matchFilter(pulses[0:1], h, metadata, useGpu=False)[0]  # (nRange, nPulses)
    idx1, width1, rel1 = _threeDbWidth(compressedLook[:, 0])
    peak1 = float(np.abs(compressedLook[idx1, 0]))
    ok1 = rel1 < 0.05
    results.append(
        (
            "1. compression works",
            ok1,
            f"peak={peak1:.5e} idx={idx1} 3dBWidth={width1} relWidth={rel1:.4f} (expect idx~987 rel<0.05)",
        )
    )

    # --- 2. Root-cause demonstration: buggy double-reversal vs correct freq MF ---
    buggy = np.correlate(sPulse0, np.conj(h[::-1]), mode="same")
    correct = np.fft.ifft(np.fft.fft(sPulse0) * np.conj(np.fft.fft(h)))
    _, wBuggy, _ = _threeDbWidth(buggy)
    _, wCorrect, _ = _threeDbWidth(correct)
    ok2 = wCorrect < wBuggy
    results.append(
        (
            "2. root-cause (double-reversal fails, freq MF compresses)",
            ok2,
            f"buggy correlate(conj(h[::-1])) width={wBuggy} vs freq-MF width={wCorrect} (expect ~19 vs ~2)",
        )
    )

    # --- 3. numpy vs cupy agreement ---
    if _HAVE_CUPY_GPU:
        cpLook = sv.matchFilter(pulses[0:1], h, metadata, useGpu=True)[0]
        denom = float(np.max(np.abs(compressedLook))) or 1.0
        relDiff = float(np.max(np.abs(compressedLook - cpLook)) / denom)
        ok3 = relDiff < 1e-9
        results.append(
            (
                "3. numpy vs cupy agree",
                ok3,
                f"maxRelDiff={relDiff:.3e} (expect <1e-9)",
            )
        )
    else:
        results.append(("3. numpy vs cupy agree", True, "SKIP (no usable GPU)"))

    # --- 4. Peak stability across pulses ---
    peakIdxs = [int(np.argmax(np.abs(compressedLook[:, p]))) for p in _STABILITY_PULSES]
    peakVals = [
        float(np.abs(compressedLook[peakIdxs[i], _STABILITY_PULSES[i]]))
        for i in range(len(_STABILITY_PULSES))
    ]
    sameIdx = len(set(peakIdxs)) == 1
    valSpread = max(peakVals) - min(peakVals)
    ok4 = sameIdx and valSpread == 0.0
    results.append(
        (
            "4. peak stable across pulses",
            ok4,
            f"peakIdxs={peakIdxs} (expect all 987) peakValSpread={valSpread:.3e} (expect 0)",
        )
    )

    # --- 5. Peak -> range (spike 1) ---
    gateStart = sv.estimateGateStart_s(compressedLook, metadata)
    fastTimeAxis = sv.buildFastTimeAxis(metadata, gateStart)
    idxPeak = int(np.argmax(np.abs(compressedLook).sum(axis=1)))
    c = float(metadata["wavePropogationSpeed_mPerSec"])
    rangeAtPeak = c * fastTimeAxis[idxPeak] / 2.0
    slantRange = float(metadata["slantRange_m"])
    withinGateRange_m = c * idxPeak / float(metadata["rxSamplingFreq_Hz"]) / 2.0
    ok5 = abs(rangeAtPeak - slantRange) < 1e-3 and 0.0 < withinGateRange_m < 250.0
    results.append(
        (
            "5. peak maps to target slant range",
            ok5,
            f"gateStart={gateStart * 1e6:.4f}us rangeAtPeak={rangeAtPeak:.3f}m "
            f"slantRange={slantRange:.3f}m withinGate={withinGateRange_m:.2f}m",
        )
    )

    # --- 6. Reference validity (lfm_ref ~ ideal up-chirp) ---
    ideal = _idealUpChirp(metadata)
    xcorr = float(np.abs(np.vdot(h, ideal)) / (np.linalg.norm(h) * np.linalg.norm(ideal)))
    ok6 = xcorr > 0.99
    results.append(
        (
            "6. lfm_ref is a valid transmit replica",
            ok6,
            f"xcorr(lfm_ref, idealUpChirp)={xcorr:.5f} (expect >0.99, ~0.9995)",
        )
    )

    # --- report ---
    print("\n=== Phase 2 matched-filter validation ===")
    allOk = True
    for name, ok, detail in results:
        allOk = allOk and ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print(f"\n=== {'ALL PASS' if allOk else 'SOME FAILED'} ===")
    return 0 if allOk else 1


if __name__ == "__main__":
    raise SystemExit(main())
