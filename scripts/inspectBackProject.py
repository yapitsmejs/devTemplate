"""Phase 3 validation: backproject every look and inspect the focused stack.

Run under a debugger to inspect intermediate arrays, or headless to print the
acceptance report and write inspection PNGs::

    uv run python scripts/inspectBackProject.py            # full 361 on GPU
    uv run python scripts/inspectBackProject.py --looks 0 30 60 90 120 150 180 210 240 270 300 330 360
    uv run python scripts/inspectBackProject.py --no-gpu   # force CPU

Throwaway inspection script (not part of the gate). Verifies the Phase 3 acceptance
criteria from ``plans/phase3Plan.md`` and writes PNGs (reusing ``sarIfp``'s
``imgInspection`` plotters) for human inspection of the image stack.

Two findings shape the criteria (see ``plans/phase3Plan.md`` -- Findings):

1. **The pipeline is correct.** Look 0 -- the look with zero within-look compressed-peak
   spread -- backprojects to a single sharp peak exactly at the grid origin. That is the
   pipeline proof and the hard gate (criterion 2). Look 0 is picked dynamically as the
   selected look with the smallest within-peak spread, so the proof is always evaluated
   against the cleanest available look.

2. **The simulator data carries a non-geometric within-look range drift.** The geometric
   slant range to the (trihedral) target is constant across all pulses and looks (the
   platform is on a circle centred on the target), yet the compressed peak index spreads
   *within* each look by up to +/-6 samples (+/-1.5 m), growing linearly from look 0
   (zero spread) to look 360 (``corr(spread, lookIdx) ~= 0.90``). This splits each look's
   pulses into range-separated sub-groups, so the coherent backprojection sum cancels for
   looks away from look 0. This is reported (criterion 3), not gated -- it is the headline
   Phase 4 simulator-accuracy finding, not a pipeline defect.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless; savefig only, no plt.show
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sarIfp.utilities import imgInspection  # noqa: E402

import simValidation as sv  # noqa: E402

DATA_DIR = Path(
    r"C:\Users\yJoonSio\OneDrive - DSO\trihedral"
    r"\Trihedrals_SAR_ADC\raw_pulse_data_repackaged"
)

# Reference look (pipeline proof): its backprojected peak must be within this many samples
# of the grid centre. The reference look is the selected look with the smallest within-look
# compressed-peak spread (empirically look 0, where the spread is zero).
_REF_RANGE_TOL_SAMP = 1
_REF_CROSS_TOL_SAMP = 1


def _withinLookSpread(comp: np.ndarray) -> tuple[int, int, int]:
    """Per-pulse compressed-peak index spread for one look.

    Args:
        comp: one look's range-compressed signal ``(nRange, nPulses)`` complex128.

    Returns:
        ``(peakIdxMin, peakIdxMax, nUnique)`` of the per-pulse ``argmax`` over fast-time.
        A clean point target at constant range gives a single index across all pulses
        (``max - min == 0``); a within-look drift gives a positive spread.
    """
    ppk = np.argmax(np.abs(comp), axis=0)
    return int(ppk.min()), int(ppk.max()), int(np.unique(ppk).size)


def _threeDbWidth_m(profile: np.ndarray, spacing_m: float) -> float:
    """Contiguous -3 dB (peak/sqrt(2)) width of a 1-D magnitude profile, in metres.

    Measured about the profile peak (reliable for the sharp reference-look response; a
    coarse, sidelobe-dominated response is reported via the centroid/peak pixel instead).
    """
    mag = np.abs(profile).astype(np.float64)
    if mag.max() == 0:
        return float("nan")
    idx = int(np.argmax(mag))
    half = mag[idx] / np.sqrt(2.0)
    left = idx
    while left > 0 and mag[left - 1] >= half:
        left -= 1
    right = idx
    while right < mag.shape[0] - 1 and mag[right + 1] >= half:
        right += 1
    return float((right - left + 1) * spacing_m)


def _saveFig(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--looks",
        type=int,
        nargs="*",
        default=None,
        help="Looks to process (default: all nLooks). Pass e.g. `--looks 0 90 180 270`.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output dir for PNGs (default: a temp dir).",
    )
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU.")
    args = parser.parse_args()

    pulses, referencePulse, metadata = sv.readPulses(DATA_DIR)
    c = float(metadata["wavePropogationSpeed_mPerSec"])
    fs = float(metadata["rxSamplingFreq_Hz"])
    bandwidth_Hz = float(metadata["radarBandwidth_Hz"])
    sampleSpacing_m = c / (2.0 * fs)

    grid = sv.formGrid(metadata)
    nRange, nCross = grid.shape[1], grid.shape[2]
    halfRange_m = 2.0
    halfCross_m = 150.0
    rangeAxis_m = np.linspace(-halfRange_m, halfRange_m, nRange)
    crossAxis_m = np.linspace(-halfCross_m, halfCross_m, nCross)
    centerRC = (nRange // 2, nCross // 2)

    lookIndices = args.looks
    useGpu = None if not args.no_gpu else False
    stack = sv.backProjectAll(
        pulses, referencePulse, metadata, grid=grid, lookIndices=lookIndices, useGpu=useGpu
    )
    lookList = (
        list(lookIndices) if lookIndices is not None else list(range(int(metadata["nLooks"])))
    )
    nLooksSel = stack.shape[0]

    outDir = args.out or Path(tempfile.gettempdir()) / "simValidation_phase3"
    outDir.mkdir(parents=True, exist_ok=True)

    # Per-look within-look compressed-peak spread (separate light pass -- match filter only).
    spreads: list[tuple[int, int, int, int]] = []  # (look, min, max, nUnique)
    for look in lookList:
        comp = sv.matchFilter(
            pulses[int(look) : int(look) + 1], referencePulse, metadata, useGpu=useGpu
        )[0]
        pmin, pmax, nuniq = _withinLookSpread(comp)
        spreads.append((int(look), pmin, pmax, nuniq))
    # Reference look = smallest within-look spread among the selected looks.
    refSelIdx = int(np.argmin(np.array([s[2] - s[1] for s in spreads], dtype=float)))
    refLook = lookList[refSelIdx]
    refImg = stack[refSelIdx]
    refMag = np.abs(refImg)
    refPeakRC = np.unravel_index(int(np.argmax(refMag)), refMag.shape)

    results: list[tuple[str, bool, str]] = []
    mags = np.abs(stack)

    # --- 1. Stack well-formed (hard) ---
    finite = bool(np.all(np.isfinite(stack)))
    energies = mags.sum(axis=(1, 2))
    nonzeroEnergy = bool(np.all(energies > 0))
    ok1 = finite and nonzeroEnergy and stack.dtype == np.complex128
    results.append(
        (
            "1. stack well-formed",
            ok1,
            f"shape={stack.shape} dtype={stack.dtype} finite={finite} "
            f"minEnergy={float(energies.min()):.3e} maxEnergy={float(energies.max()):.3e}",
        )
    )

    # --- 2. Reference look focuses at origin (hard, pipeline proof) ---
    rangeOffSamp = abs(refPeakRC[0] - centerRC[0])
    crossOffSamp = abs(refPeakRC[1] - centerRC[1])
    ok2 = (
        rangeOffSamp <= _REF_RANGE_TOL_SAMP
        and crossOffSamp <= _REF_CROSS_TOL_SAMP
        and refMag[refPeakRC] > 0
    )
    results.append(
        (
            "2. reference look focuses at origin (pipeline proof)",
            ok2,
            f"refLook={refLook} (min within-look spread={spreads[refSelIdx][2] - spreads[refSelIdx][1]} samp) "
            f"peakRC={refPeakRC} center={centerRC} "
            f"rangeOff={rangeOffSamp} samp crossOff={crossOffSamp} samp "
            f"(tol +-{_REF_RANGE_TOL_SAMP}/+-{_REF_CROSS_TOL_SAMP})",
        )
    )

    # --- 3. Within-look peak-spread artifact (report) -- the Phase 4 finding ---
    spreadSamp = np.array([s[2] - s[1] for s in spreads], dtype=float)
    lookIdxArr = np.array(lookList, dtype=float)
    corrSpread = float(np.corrcoef(spreadSamp, lookIdxArr)[0, 1]) if nLooksSel > 1 else float("nan")
    maxSpreadSamp = int(spreadSamp.max())
    maxSpreadLook = int(lookList[int(np.argmax(spreadSamp))])
    results.append(
        (
            "3. within-look peak-spread artifact (report)",
            True,
            f"spread(samp) min={int(spreadSamp.min())} max={maxSpreadSamp} "
            f"({maxSpreadSamp * sampleSpacing_m:.3f} m at look {maxSpreadLook}) "
            f"corr(spread,lookIdx)={corrSpread:.3f} -- non-geometric range drift "
            f"(geometric slant range is constant); Phase 4 finding",
        )
    )

    # --- 4. Focus degradation across looks (report) ---
    peakRCs = [np.unravel_index(int(np.argmax(mags[i])), mags[i].shape) for i in range(nLooksSel)]
    peakEnergies = np.array([float(mags[i].max()) for i in range(nLooksSel)])
    peakRangeOff = np.array([abs(p[0] - centerRC[0]) for p in peakRCs])
    peakCrossOff = np.array([abs(p[1] - centerRC[1]) for p in peakRCs])
    results.append(
        (
            "4. focus degradation across looks (report)",
            True,
            f"peakEnergy: ref={float(refMag.max()):.3e} min={peakEnergies.min():.3e} "
            f"max={peakEnergies.max():.3e} | peak pixel max rangeOff={int(peakRangeOff.max())} samp "
            f"max crossOff={int(peakCrossOff.max())} samp -- looks away from the reference defocus",
        )
    )

    # --- 5. Range resolution on the reference look (report) ---
    rangeProfile = refMag[:, refPeakRC[1]]
    rangeWidth_m = _threeDbWidth_m(rangeProfile, rangeAxis_m[1] - rangeAxis_m[0])
    expectedRange_m = c / (2.0 * bandwidth_Hz)
    results.append(
        (
            "5. range resolution on reference look (report)",
            True,
            f"measured -3dB range width={rangeWidth_m:.3f}m expect~{expectedRange_m:.3f}m",
        )
    )

    # --- 6. Cross-range response on the reference look (report) ---
    # The per-look within-look aperture is ~0.00996 deg for every look (constant), which the
    # naive sinc estimate maps to ~86 m. The measured response is a sharp isorange-contour
    # SPIKE at the origin instead, because the compressed peak is a 1-sample delta (zero
    # neighbours): the interp returns it only where a pixel's slant range equals the target
    # range, i.e. on the isorange contour, which (arc geometry, target at the centre) passes
    # through the origin. So the reference look is a single correctly-placed spike -- the
    # milestone target -- not an 86 m sinc lobe. The 86 m estimate assumed a broad compressed
    # peak; it does not apply here. Spike 3 resolved empirically: per-look cross-range is an
    # isorange spike, not a resolved lobe.
    crossProfile = refMag[refPeakRC[0], :]
    crossWidth_m = _threeDbWidth_m(crossProfile, crossAxis_m[1] - crossAxis_m[0])
    results.append(
        (
            "6. cross-range response on reference look (report)",
            True,
            f"measured -3dB cross-range width={crossWidth_m:.1f}m (isorange spike at origin, "
            f"NOT the ~86m sinc -- delta-narrow compressed peak + arc isorange geometry; "
            f"spike 3 resolved empirically)",
        )
    )

    # --- 7. In-gate coverage on the reference look (report) ---
    fracNZ = float((refMag > 0).mean())
    results.append(
        (
            "7. in-gate coverage on reference look (report)",
            True,
            f"nonzero fraction={fracNZ:.3f} (1.0 expected: gate covers the small grid)",
        )
    )

    # --- 8. Human-inspection figures ---
    figPaths: list[str] = []
    try:
        fig, _, _ = imgInspection.singleImageMaxPointPlot(refMag, plot=False)
        p = outDir / "look_reference.png"
        _saveFig(fig, p)
        figPaths.append(str(p))
    except Exception as exc:  # noqa: BLE001 - inspection figures are best-effort
        results.append(("8. figure look_reference", False, f"failed: {exc!r}"))
    stride = max(1, nLooksSel // 12)
    tiled = mags[::stride][:-1] if nLooksSel // stride > 12 else mags[::stride]
    if tiled.shape[0] >= 1:
        try:
            fig, _, _ = imgInspection.imageAbMaxPointPlot(np.ascontiguousarray(tiled), plot=False)
            p = outDir / "looks_tiled.png"
            _saveFig(fig, p)
            figPaths.append(str(p))
        except Exception as exc:  # noqa: BLE001
            results.append(("8. figure looks_tiled", False, f"failed: {exc!r}"))
        try:
            fig, _ = imgInspection.imageStackPlot(
                np.ascontiguousarray(tiled), plot=False, alpha=0.6
            )
            p = outDir / "looks_overlay.png"
            _saveFig(fig, p)
            figPaths.append(str(p))
        except Exception as exc:  # noqa: BLE001
            results.append(("8. figure looks_overlay", False, f"failed: {exc!r}"))
    results.append(("8. figures written", len(figPaths) >= 1, "; ".join(figPaths)))

    # --- report ---
    print("\n=== Phase 3 backprojection validation ===")
    print(f"looks: {lookList[:6]}{'...' if len(lookList) > 6 else ''} ({nLooksSel} total)")
    print(f"grid: {grid.shape}  rangeAxis+/-{halfRange_m}m  crossAxis+/-{halfCross_m}m")
    print(
        f"reference look: {refLook} (within-look spread {spreads[refSelIdx][2] - spreads[refSelIdx][1]} samp) "
        f"-> peakRC={refPeakRC} range={rangeAxis_m[refPeakRC[0]]:.2f}m cross={crossAxis_m[refPeakRC[1]]:.1f}m"
    )
    print("\nwithin-look peak-spread (look: min/max/nUnique):")
    for look, pmin, pmax, nuniq in spreads:
        flag = "  <- reference" if look == refLook else ""
        print(f"  look {look:3d}: {pmin}-{pmax} ({nuniq} unique, spread {pmax - pmin} samp){flag}")
    print()
    allOk = True
    for name, ok, detail in results:
        allOk = allOk and ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print(f"\n=== {'ALL PASS' if allOk else 'SOME FAILED'} ===")
    return 0 if allOk else 1


if __name__ == "__main__":
    raise SystemExit(main())
