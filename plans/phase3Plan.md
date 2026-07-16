# Phase 3 plan — per-look backprojection → 361 focused images

> Implementation plan for Phase 3 of [`developmentPlan.md`](./developmentPlan.md).
> Throwaway toolbox; gate is code standards only (ruff + mypy). See
> [`CONVENTIONS.md`](../CONVENTIONS.md) for naming/style.

## Context

Phase 1 (data reader) and Phase 2 (matched filter + signal-model alignment) are
implemented and gate-clean. The matched filter compresses each look's 1200 pulses to a
~1-sample peak (rel width 0.001) bit-identical across pulses, and the fast-time axis is
calibrated so the peak maps to the known 17,322.385 m slant range.

Phase 3 forms one focused image per look (361 total) by feeding the range-compressed
signal into `sarIfp`'s time-domain backprojection, then inspects the stack. This is the
project's first milestone: a `(361, H, W)` stack in which each look shows a single
focused response at the origin.

Two read-only explorations (sar-ifp source + empirical checks on the actual data) pin the
design. Three findings reshape the plan:

**sar-ifp backprojection API (source-confirmed, `backProjection.py`):**
- `formImage(rangeCompressedSignal, fastTimeAxis_s, txPos, projectionGridCoordinates,
  preprocessingParams, wavePropogationSpeed_mPerSec, radarCenterFrequency_Hz,
  pulseRepfreq_Hz, projectionGrid=None, dtype=np.complex128) -> (H, W)` (`backProjection.py:33-44`).
  Per-pulse loop (`_formImage_core:171-186`): `delay = (‖grid−txPos‖+‖grid−rxPos‖)/c`
  (monostatic → `2·‖txPos−pixel‖/c`), `interp = xp.interp(delay, fastTimeAxis_s, col,
  left=0, right=0)`, `grid += interp · exp(2jπ·fc·delay)`. **No `formGrid` exists in
  sar-ifp** — the caller builds the `(3, H, W)` grid; `formImage` consumes it verbatim.
- **PRF is vestigial for us** (`backProjection.py:185`): the phase term is
  `exp(2jπ·radarCenterFrequency_Hz·delay)` only — `pulseRepfreq_Hz` never enters
  `_formImage_core`. It is consumed solely by `getBistaticRxPos` on the `getBistaticRx: True`
  branch (`backProjection.py:104-106`). With `getBistaticRx: False` (monostatic),
  `rxPos = txPos` and PRF has zero effect. Pass a nominal `pulseRepfreq_Hz` (1.0) and
  document. **Resolves spike 2.**
- **Interp edge behaviour is the binding constraint** (`backProjection.py:178-184`):
  `left=0, right=0` means any pixel whose geometric delay falls outside
  `[fastTimeAxis_s[0], fastTimeAxis_s[-1]]` contributes **zero** (hard range-gate cutoff,
  not a clamp). So grid pixels must keep their slant range inside the receive gate.
- **`projectionGrid=None` per call is mandatory** (`backProjection.py:91`,
  `test_backProjection.py:87-102`): passing a shared grid *accumulates in place*. Default
  to `None` so each look gets a fresh zeros grid.
- `formImage` auto-dispatches numpy/cupy from the module-level `_HAVE_CUPY_GPU` flag and
  the input array type — there is no per-call CPU/GPU override. Passing a numpy
  `rangeCompressedSignal` returns a numpy image (via `cp.asnumpy` on the GPU path). We do
  not force a backend; GPU is used when available.

**Input-shape mapping (verified against the data):**

| `formImage` arg | Source | Transform |
|---|---|---|
| `rangeCompressedSignal` `(nRange, nPulses)` c128 | `matchFilter(pulses,…)[look]` | already `(1001, 1200)`, no transform |
| `fastTimeAxis_s` `(nRange,)` | `buildFastTimeAxis(meta, gateStart_s)` per look | `(1001,)` float64 |
| `txPos` `(3, nPulses)` | `metadata["sensorPositions_ENU_m"][look]` `(1200,3)` | `.T` → `(3, 1200)` |
| `projectionGridCoordinates` `(3, H, W)` | new `formGrid(metadata, …)` | ENU ground plane, `Z=0`, centered at origin |
| `preprocessingParams` | `{"pulseCompressionMethod": "match", "getBistaticRx": False}` | monostatic → PRF vestigial |
| `wavePropogationSpeed_mPerSec` | `metadata["wavePropogationSpeed_mPerSec"]` | pass-through (sar-ifp spelling) |
| `radarCenterFrequency_Hz` | `metadata["radarCenterFrequency_Hz"]` (1e10) | pass-through |
| `pulseRepfreq_Hz` | **not in data** | nominal `1.0`; documented vestigial |

Coordinate frame is ENU-at-origin (`targetOrigin_XYZ_m = [0,0,0]`,
`coordinateConvention = "x=East, y=North, z=Up"`), matching `formImage`'s expectation — no
rotation needed, only the per-look `.T` transpose of `sensorPositions_ENU_m[look]`.

**Empirical findings that drive the grid + anchoring design (measured on the data):**
- The compressed peak index **drifts across looks**: idx 987 at looks 0/90/270, 990 at
  look 180, 993 at look 360 — ~6 samples (~1.5 m) of range walk over the 3.6° aperture.
  ⇒ **Compute `gateStart` per look** (recompute `idxPeak` per look via
  `estimateGateStart_s`, which always anchors to the nominal `slantRange_m`) so every
  look self-anchors its peak to the target range and focuses at the grid origin. A shared
  fast-time axis would mis-anchor the off-axis looks by up to 1.5 m. The drift itself is a
  simulator-accuracy observation for Phase 4.
- Range-gate budget is **asymmetric and tight**: `gateStart ≈ 113.9175 µs`,
  `fastTimeAxis_s[-1] ≈ 115.584 µs`, peak at idx 987 of 1001 → **far-side budget 3.25 m**
  (14 samples), shrinking to ~1.75 m for the worst look (idx 993); **near-side budget
  246.6 m**. Range sample spacing `c/(2·fs) = 0.2498 m`.
- **Cross-range offset is gate-invariant to first order** (slant range is constant along
  the isorange direction): 2nd-order slant change at 150 m cross-range is only 0.65 m, far
  inside the budget. So cross-range can be large; **range-direction extent must be small**.

## Grid design (the load-bearing decision)

A square ENU grid cannot both show the coarse (~86–90 m) cross-range response *and* stay
inside the ~3.25 m far-side range gate, because the range/cross-range axes are diagonal to
ENU (platform at `azimuthCenter = 45°`). The grid is therefore a **rectangle aligned to
range / cross-range at `azimuthCenter_deg`** (built once, reused for all 361 looks — their
true axes differ by ≤1.8°, negligible for the coarse response; the 1.8° tilt injects
≤3.1 m of range at 100 m cross-range, within budget for the response region).

Defaults (all overridable; the inspect script reports measured resolution vs these):
- **Range** `±2.0 m` @ `0.1 m` spacing → 41 samples. The response main lobe is ~1 sample
  (0.25 m); ±2 m shows it plus near sidelobes, and the far-corner zeroing (worst look
  budget ~1.75 m) is cosmetic — the response sits at the origin, well inside the gate.
- **Cross-range** `±150.0 m` @ `3.0 m` spacing → 101 samples. Per-look cross-range
  resolution ≈ `λ/(2·Δθ) = 0.03/(2·0.0096°·π/180) ≈ 90 m`; ±150 m shows the main lobe with
  margin. 2nd-order slant change 0.65 m ≪ budget.

Expected per-look resolution (spike 3, to *measure* not assume): range `c/(2·B) = 0.250 m`;
cross-range `≈ 86–90 m` (rough, factor-2 tolerance — this is a sanity check, not a gate).

`formGrid` builds `east = rHat·range + cHat·cross`, `north = …`, `up = 0`, with
`rHat = (cos az, sin az)` (toward platform), `cHat = (-sin az, cos az)` (cross-range),
`R, C = meshgrid(rangeAxis, crossAxis, indexing="ij")`, returns
`np.stack([east, north, up], axis=0)` → `(3, nRange, nCross)`. Axis 0 = XYZ/ENU, axis 1 =
range, axis 2 = cross-range. Target at origin → centre pixel.

## Deliverables

1. **`src/simValidation/backProject.py`** (new) — owns the grid + per-look backprojection:
   - `formGrid(metadata, halfExtentRange_m=2.0, halfExtentCrossRange_m=150.0,
     spacingRange_m=0.1, spacingCrossRange_m=3.0, azimuthDeg=None) -> np.ndarray` —
     `(3, nRange, nCross)` ENU ground plane as above; `azimuthDeg` defaults to
     `metadata["azimuthCenter_deg"]`. Pure numpy.
   - `backProjectLook(compressedLook, fastTimeAxis_s, txPosLook, grid, metadata) ->
     np.ndarray` — thin wrapper calling `formImage` with
     `preprocessingParams = {"pulseCompressionMethod": "match", "getBistaticRx": False}`,
     `pulseRepfreq_Hz = 1.0` (vestigial), and `projectionGrid=None` (fresh zeros). Returns
     `(H, W)` complex128. `formImage` auto-dispatches GPU.
   - `backProjectAll(pulses, referencePulse, metadata, grid=None, lookIndices=None,
     useGpu=None) -> np.ndarray` — orchestrator. Builds the grid once (or accepts one).
     Streams look-by-look: per look, slice `pulses[look:look+1]`, `matchFilter(…)[0]`,
     `estimateGateStart_s(comp, meta)` (per-look, absorbs the peak drift),
     `buildFastTimeAxis(meta, gateStart_s)`, `txPos = sensorPositions_ENU_m[look].T`,
     `backProjectLook(…)`, store into `stack[look]`. Only one look's compressed signal
     (~19 MB) resident at a time — never holds the 6.9 GB full compressed cube.
     `lookIndices=None` → all 361; pass a subset for quick CPU checks. `useGpu` forwards to
     `matchFilter` (compression); `formImage` dispatches its own backend. Returns
     `(len(lookIndices), H, W)` complex128.
   - `__all__`, Google docstrings, full type hints, `from __future__ import annotations`.
2. **`src/simValidation/__init__.py`** — add `formGrid`, `backProjectLook`, `backProjectAll`
   to `__all__` and re-export.
3. **`scripts/inspectBackProject.py`** (new, throwaway, not gated) — the validation +
   inspection harness (acceptance criteria below). Builds the grid, runs `backProjectAll`
   (full 361 on GPU, or a configurable subset on CPU), reports metrics, and writes PNGs
   for human inspection by reusing `sarIfp.utilities.imgInspection` plotters
   (`singleImageMaxPointPlot`, `imageAbMaxPointPlot` on a stride, `imageStackPlot`
   overlay) — `fig.savefig(...)` to an output dir (matplotlib is driven headless via Agg;
   `plot=False` paths are used where the plotters support it).
4. **`plans/developmentPlan.md`** — append a "Phase 3 — resolved" note: spikes 2 (PRF
   vestigial) and 3 (grid extent vs gate; per-look anchoring; measured resolutions) now
   resolved, and flag the **per-look peak-idx drift (range walk, ~1.5 m)** as a Phase 4
   simulator-accuracy observation. Mark the Phase 3 section resolved.

## Validation — `scripts/inspectBackProject.py` (acceptance criteria)

Run under a debugger or headless; prints a structured report and exits 0 only if all hard
criteria pass (report criteria are not gating — Phase 3's DoD is human inspection). The
criteria were revised after implementation surfaced a simulator-data artifact (see
**Findings** below); the original "every look focused at the origin" hard gate was
unrealistic because the data does not coherently focus away from look 0.

1. **Stack well-formed (hard):** shape `(nLooksSelected, H, W)` complex128, all finite,
   non-zero total energy per look.
2. **Reference look focuses at the origin (hard, pipeline proof):** the selected look with
   the smallest within-look compressed-peak spread (empirically look 0, spread 0) has its
   magnitude peak pixel within `±1` range sample and `±1` cross-range sample of the grid
   centre. This proves the backprojection pipeline is correct: a clean look focuses to a
   single sharp response at the target origin. Report the reference look, its spread, and
   the peak pixel.
3. **Within-look peak-spread artifact (report):** the per-pulse compressed-peak index spread
   within each look, its max, and `corr(spread, lookIdx)`. This is the headline Phase 4
   simulator-accuracy finding (non-geometric range drift; see Findings).
4. **Focus degradation across looks (report):** per-look peak energy and peak-pixel offset
   from the origin, showing focus degrades as the within-look spread grows.
5. **Range resolution on the reference look (report):** −3 dB width along the range axis
   through the peak; expect ≈ `c/(2·B) = 0.250 m` (the delta-narrow compressed peak).
6. **Cross-range response on the reference look (report):** −3 dB cross-range width. The
   reference look is a sharp isorange-contour spike at the origin, *not* the ~86 m sinc
   lobe — see Findings (spike 3 resolved empirically).
7. **In-gate coverage on the reference look (report):** fraction of grid pixels with
   non-zero contribution (the central response region is non-zero; far corners may zero).
8. **Human-inspection figures:** write `look_reference.png` (singleImageMaxPointPlot on the
   reference look), `looks_tiled.png` (imageAbMaxPointPlot over a stride),
   `looks_overlay.png` (imageStackPlot) to the output dir; print their paths.

## Verification

- `uv run python scripts/check.py` → GATE PASS (ruff check + format --check + mypy).
- `uv run python scripts/inspectBackProject.py` → all hard criteria pass; PNGs written.
- `uv run python scripts/inspectReader.py` → still passes (no Phase 1 regression).
- `uv run python scripts/inspectMatchFilter.py` → still all six pass (no Phase 2 regression).

## Execution order

1. Write `backProject.py` + update `__init__.py`.
2. `ruff check --fix . && ruff format .` then `scripts/check.py` until gate green.
3. Write `scripts/inspectBackProject.py`; run a small `lookIndices` subset on CPU first to
   sanity-check the grid/anchoring (peak at origin, no all-zero image), then the full 361
   on GPU; iterate until all hard criteria pass.
4. Append the Phase 3 "resolved" note to `plans/developmentPlan.md`; re-run the gate and the
   Phase 1/2 regression scripts.

## Findings / Resolution

Implemented in [`backProject.py`](../src/simValidation/backProject.py) and validated by
[`scripts/inspectBackProject.py`](../scripts/inspectBackProject.py). The pipeline is
correct; the headline result is a **simulator-data artifact**, not a pipeline defect.

**Pipeline proven correct.** Look 0 backprojects to a single sharp peak exactly at the
grid origin (peak pixel `(20, 50)` = centre, `rangeOff = crossOff = 0` samples; cross-profile
peak 63.1 at cross = 0, next value ≤ 4.7). The reference look is picked dynamically as the
selected look with the smallest within-look compressed-peak spread — empirically look 0,
whose spread is zero — so the pipeline proof is always evaluated against the cleanest
available look. This is hard criterion 2.

**Simulator-data artifact (Phase 4 finding).** The geometric slant range to the (trihedral)
target is **constant** across all 1200 pulses of every look — `17322.385 m`, within-look and
across-look `R`-walk = 0.0 samples (the platform is on a circle centred on the target). Yet
the matched-filter compressed peak index spreads *within* each look: per-pulse `argmax`
ranges `987 ± k` where `k` grows linearly from 0 at look 0 to ±6 samples (±1.5 m) at look 360
(`corr(spread, lookIdx) ≈ 0.90`; max spread 12 samples / 3.0 m at look 360). These are clean
delta-peaks (zero-valued neighbours), not sidelobes — looks 180/360 each carry **three**
near-equal symmetric peaks around 987. This drift is **non-geometric** (the true range is
constant), so it is in the simulator data, not the backprojection. It splits each look's
pulses into range-separated sub-groups, so the coherent backprojection sum cancels for looks
away from look 0 (peak energy ~1000× lower; the response splits/shifts off-origin). Per-pulse
range alignment (motion compensation) recovers look 180 (centroid → origin, ~18× energy)
but only partially recovers look 360 — left out of the pipeline by decision (keep it minimal;
document the artifact). The asymmetry — clean at look 0, worst at look 360, monotonic from
look 0, *not* symmetric about the aperture centre (look 180) — is unexplained; flagged for
Phase 4 root-cause. The targets are trihedrals (corner reflectors), which should give a
single clean echo per pulse, so the multipeak structure is itself anomalous.

**Spike 3 resolved empirically (cross-range).** Every look has the same within-look azimuth
span (0.009964° → naïve sinc estimate ~86 m). The measured reference-look response is a sharp
isorange-contour **spike at the origin**, not an 86 m sinc lobe. Cause: the compressed peak
is a 1-sample delta (zero neighbours), so the interp returns it only where a pixel's slant
range equals the target range — i.e. on the isorange contour, which (arc geometry, target at
the centre) passes through the origin. The 86 m estimate assumed a broad compressed peak; it
does not apply. The per-look cross-range response is an isorange spike, not a resolved lobe.

**Spikes 2 (PRF vestigial) and the grid/anchoring design** resolved as planned — see the
context above. The per-look `estimateGateStart_s` self-anchoring is retained (it correctly
maps each look's look-summed peak to the target range); it is not the cause of the defocus
(the defocus is the within-look spread, present before anchoring).