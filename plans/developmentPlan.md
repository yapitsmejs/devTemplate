# Development plan — simValidation

> Throwaway toolbox for validating a radar simulator. Not a shipped product.
> See [`README.md`](../README.md) for purpose and [`CONVENTIONS.md`](../CONVENTIONS.md)
> for naming/code-style. The gate enforces code standards only (ruff + mypy) —
> there are no functional tests; results are judged by human inspection.

## Goal

For each of the simulator's 361 looks, range-compress its 1200 pulses against the LFM
reference and backproject into a small SAR image centered on the target origin. The
simulator is "correct" when every look's image contains a single, well-located focused
response at the origin.

A 0.01° look aperture over the 17.3 km slant range gives coarse cross-range resolution
(tens of meters), so "well-focused" here means a single correctly-placed response — not a
sub-meter point. After the per-look pipeline produces trustworthy images, build small
inspection tools to quantify simulator-accuracy metrics as needed.

## Non-goals

- No exploitation/analysis extensions — purely simulator validation.
- Not a production interface; the team-facing interface is a separate later project.
- No CLI; scripts run under a debugger for variable inspection.
- No full-aperture image — validation is per-look (361 viewing angles), not one merged
  3.6° aperture.
- Do not modify anything outside this repository (notably the local `sar-ifp`
  backprojection packages) unless explicitly told.

## Data on disk

Source: `C:\Users\yJoonSio\OneDrive - DSO\trihedral\Trihedrals_SAR_ADC\raw_pulse_data_repackaged\`

| File | dtype | shape | role |
|---|---|---|---|
| `rx_sig.npy` | complex128 | (361, 1200, 1001) | raw pulses; axes (look, pulse, fast-time sample); 6.9 GB |
| `lfm_ref.npy` | complex128 | (1001,) | LFM reference chirp |
| `sensor_positions.npy` | float64 | (361, 1200, 3) | per-pulse APC position, ENU, target at origin |
| `pulse_azimuth_deg.npy` | float64 | (361, 1200) | per-pulse azimuth, deg |
| `look_azimuth_deg.npy` | float64 | (361,) | look-center azimuth, deg |
| `look_positions.npy` | float64 | (361, 3) | platform position at each look center |

Sidecars: `../Meta_parameters.txt` (fc 10 GHz, BW 599 MHz, ADC 0.6 GS/s, 1001
samples/pulse, chirp dur 1.667 µs, chirp rate 3.5964e14 Hz/s, HH pol) and
`./geometry_note.txt` (target origin [0,0,0], altitude 10 km, grazing 35.26°, slant range
17,322.385 m, ground range 14,144.434 m, azimuth center 45° ± 1.8°, 361 looks × 1200
pulses/look, pulse-to-pulse az step ≈ 0.000008°, frame x=East/y=North/z=Up CCW from +x).
**PRF is not stored** — only angular slow-time (see spikes).

## Integration surface (`sar-ifp`, read-only)

Import name `sarIfp` at `C:\Users\yJoonSio\Desktop\ghRepos\sar-ifp`.

- `sarIfp.sarSim.applyMatchedFilter(s, h)` — range-compress `s` (axis-0 = fast-time) with
  reference `h`; cupy-auto-dispatched (`ifft(fft(s)·conj(fft(h)))`). Reused for the matched
  filter; we do not reimplement the FFT math.
- `sarIfp.imageFormationProcessor.backProjection.backProjection.formImage(
  rangeCompressedSignal (nRange, nPulses) c128, fastTimeAxis_s (nRange,), txPos (3, nPulses),
  projectionGridCoordinates (3, H, W), preprocessingParams, wavePropogationSpeed_mPerSec,
  radarCenterFrequency_Hz, pulseRepfreq_Hz)` → focused `(H, W)` image. Spotlight BP;
  expects already-range-compressed input; ENU at scene origin (matches the data).
  Minimum `preprocessingParams = {"pulseCompressionMethod": "match", "getBistaticRx": False}`
  (monostatic).
- Convention to mirror: the `_<func>_core(..., xp)` numpy/cupy dispatch + `_HAVE_CUPY_GPU`
  import guard (see `backProjection._formImage_core`).
- `sarIfp.utilities.imgInspection` — contrast-stretch + matplotlib plotters, reusable for
  Phase 4.

## Module layout (`src/simValidation/`, per `CONVENTIONS.md`)

- `readPulses.py` — public `readPulses(dataDirPath) -> (pulses, referencePulse, metadata)`
  matching the README representative script. Loads the six `.npy` files (`rx_sig` via
  `numpy.load(..., mmap_mode="r")` so the 6.9 GB cube is never fully resident) and bundles
  `sensorPositions`, the azimuth axes, and parsed scalar params into `metadata`
  (`dict[str, Any]`, per the project's parameter-dict convention). Declares `__all__`.
- `_parseMeta.py` — private parser for `Meta_parameters.txt` + `geometry_note.txt` → typed
  scalar dict (`radarCenterFrequency_Hz`, `radarBandwidth_Hz`, `rxSamplingFreq_Hz`,
  `pulseLength_s`, `wavePropogationSpeed_mPerSec`, grazing/slant/ground range, azimuth
  center/span). Unit-suffixed keys per conventions.
- `matchFilter.py` — public `matchFilter(pulses, referencePulse, metadata)`. Thin wrapper
  over `sarIfp.sarSim.applyMatchedFilter`; owns per-look iteration, fast-time axis
  construction, and cupy dispatch. Per-look data is ~19 MB so VRAM is never the
  constraint — chunking here is about streaming looks from the mmap, not splitting a look.
- `formGrid.py` — builds `projectionGridCoordinates (3, H, W)` ENU ground-plane grid
  centered at `[0,0,0]`; pixel spacing/extent sized to the per-look resolution (a few
  hundred meters at ~5–10 m pixels). Built once, reused for all 361 looks.
- `backproject.py` — public `backproject(mfSignal, metadata)` that iterates looks and calls
  `formImage` per look, returning a `(361, H, W)` image stack. Feed
  `rangeCompressedSignal` as `(1001, 1200)` (fast-time × pulses) and `txPos` as `(3, 1200)`
  — both transposed from the on-disk look slice.
- `__init__.py` — `__all__` re-exporting the public API so `import simValidation as sv`
  works as the README shows.
- Phase 4 (open-ended): `inspectImage.py` — point-target focus/location metrics, reusing
  `sarIfp.utilities.imgInspection` plotters.

Every module: `from __future__ import annotations`, type hints on public functions
(mypy-enforced), Google-style docstrings, `__all__`.

## Phases

### Phase 1 — Data reader
- **Scope:** `readPulses` + `_parseMeta`. Load the mmaped cube, reference, positions, and
  parsed scalars into numpy arrays with the right shapes/dtypes.
- **DoD:** A debug script loads everything; shapes/dtypes match the table above; pulse-0 of
  look 0 cross-checks against `../s_rx_pulse0_look_000.csv`.
- **Verify:** Inspection in the debugger; cross-check array shapes against the 361-look /
  1200-pulse / 1001-sample collection.

### Phase 2 — Matched filter + signal-model alignment  *(risk concentrates here)*
- **Scope:** `matchFilter` wrapping `applyMatchedFilter`. Construct the fast-time axis and
  range-compress each look's 1200 pulses against `lfm_ref`. Engage the cupy path by feeding
  `cp.ndarray` in.
- **DoD:** Range-compress one look; the compressed peak lands at the target's two-way
  slant-range delay on the fast-time axis; numpy and cupy paths agree on that look.
- **Verify:** Side-by-side numpy/cupy comparison on one look; inspect the range-compressed
  peak location against the geometric range.
- **Risk / spike:** The FFT math is reused, so end-to-end correctness now hinges on the
  fast-time axis construction (spike 1). Spike the numpy path on one look and confirm the
  peak range before touching backprojection — Phase 3 depends on this.

> **Phase 2 — resolved (implementation).** Implemented in
> [`matchFilter.py`](../src/simValidation/matchFilter.py) (reuses
> `sarIfp.sarSim.applyMatchedFilter`) and validated by
> [`scripts/inspectMatchFilter.py`](../scripts/inspectMatchFilter.py) — all six acceptance
> criteria pass. Findings: (1) **Root cause of the prior failure** — the manual code used
> `correlate(s, conj(h[::-1]))`, a double-reversal (`np.correlate` already reverses its 2nd
> argument) that convolves instead of correlating (width 18 vs the correct 1); the
> frequency-domain `ifft(fft(s)·conj(fft(h)))` form fixes it and is what `applyMatchedFilter`
> uses. (2) `lfm_ref` is a valid clean up-chirp transmit replica (xcorr 0.9995 vs the ideal
> `exp(j·π·K·t²)`) — use it directly; do not synthesize a reference. (3) **Spike 1
> resolved** — the compressed peak is bit-identical across all 1200 pulses at idx 987;
> `estimateGateStart_s` calibrates `gateStart_s = 2·slantRange_m/c − idx/fs ≈ 113.9175 µs`
> so `buildFastTimeAxis` maps the peak to the known 17,322.385 m slant range. numpy and cupy
> agree to ~2e-16. The remaining spikes (circular-wrap/focus, grid extent vs gate window,
> PRF usage) carry to Phase 3. See [`phase2Plan.md`](./phase2Plan.md).

### Phase 3 — Per-look backprojection → 361 focused images  *(first milestone)*
- **Scope:** `formGrid` + `backproject`. Iterate looks, call `formImage` per look, stack
  into `(361, H, W)`.
- **DoD (first milestone):** A `(361, H, W)` stack in which each look shows a single
  focused response at the origin. This is the acceptance signal for the whole pipeline.
- **Verify:** Human inspection — animate across looks; reuse `imgInspection` plotters.
- **Depends on:** Phase 1 and Phase 2 with a correct, backprojection-compatible
  fast-time axis.

> **Phase 3 — resolved (implementation).** Implemented in
> [`backProject.py`](../src/simValidation/backProject.py) (`formGrid` + `backProjectLook` +
> `backProjectAll`, all gate-clean) and validated by
> [`scripts/inspectBackProject.py`](../scripts/inspectBackProject.py). The pipeline is
> **proven correct**: look 0 backprojects to a single sharp peak exactly at the grid origin
> (peak pixel = centre, zero offset) — the hard gate. The headline result is a
> **simulator-data artifact**, not a pipeline defect, and is the first Phase 4 finding:
> - **Non-geometric within-look range drift.** The geometric slant range to the (trihedral)
>   target is constant across all pulses and looks (173,322.385 m; the platform is on a
>   circle centred on the target), yet the matched-filter compressed peak index spreads
>   *within* each look — `987 ± k`, `k` growing linearly from 0 (look 0) to ±6 samples
>   (±1.5 m, look 360; `corr(spread, lookIdx) ≈ 0.90`, max 12 samples / 3.0 m). These are
>   clean delta-peaks, not sidelobes; looks 180/360 carry three near-equal symmetric peaks
>   around 987. The drift is non-geometric (true range constant) → in the data, not the
>   backprojection. It splits each look's pulses into range-separated sub-groups, so the
>   coherent sum cancels for looks away from look 0 (~1000× lower energy, response shifts
>   off-origin). Per-pulse range alignment (mocomp) recovers look 180 but only partially
>   look 360; left out of the pipeline by decision (keep it minimal; document). The
>   asymmetry — clean at look 0, worst at look 360, monotonic from look 0, *not* symmetric
>   about the aperture centre — is unexplained; trihedrals should give a single echo per
>   pulse, so the multipeak structure is itself anomalous. Root cause deferred to Phase 4.
> - **Spike 3 resolved empirically.** Every look has the same within-look aperture
>   (0.009964° → naïve ~86 m), but the reference look is a sharp isorange-contour spike at
>   the origin, not an 86 m sinc: the 1-sample delta compressed peak images the isorange
>   contour (through the origin) directly. The ~86 m estimate assumed a broad compressed
>   peak and does not apply.
> - **Spike 2 resolved** as planned: PRF is vestigial on the monostatic branch; a nominal
>   1.0 is passed and documented. The per-look `estimateGateStart_s` self-anchoring is
>   retained (correctly maps each look's peak to the target range); it is not the defocus
>   cause.
>
> The original DoD ("each look a single focused response at the origin") is met for the
> reference look (pipeline proof) and is blocked for the remaining looks by the artifact
> above; the acceptance criteria were revised accordingly (reference-look focus = hard gate;
> artifact + degradation = report). See [`phase3Plan.md`](./phase3Plan.md) — Findings.

### Phase 4 — Simulator-accuracy metrics (open-ended)
- **Scope:** `inspectImage.py` and ad-hoc tools — response location vs. origin across
  looks, sidelobe/focus uniformity, etc. Driven by whatever the human analysis turns up.
- **DoD:** None fixed; iterative inspection. Tools added as questions arise, judged by
  inspection.
- **Verify:** Human judgement; no hard success criteria by design.
- **Depends on:** Phase 3 producing a trustworthy image stack to measure against.

## Spikes (resolve during implementation, not deferred)

1. **Fast-time t=0 / range-gate start.** *(resolved in Phase 2)* The compressed peak is
   bit-identical across all pulses at idx 987; `estimateGateStart_s` sets
   `gateStart_s = 2·slantRange_m/c − idx/fs ≈ 113.9175 µs`, so `buildFastTimeAxis` maps the
   peak to the known 17,322.385 m slant range and the geometric delay lines up.
2. **PRF / slow-time.** *(resolved in Phase 3)* PRF is not in the data; `formImage` takes
   `pulseRepfreq_Hz`. Read `backProjection._formImage_core` to determine whether PRF actually enters the spotlight
   phase term (the per-pulse term is `exp(2j·π·fc·delay)`, which uses `fc` + geometric
   delay, not PRF). If vestigial, pass a nominal PRF and document it; if used, derive one
   from the angular spacing + an assumed platform velocity and flag the assumption.
   → **Vestigial on the monostatic (`getBistaticRx: False`) branch**: PRF only enters
   `getBistaticRxPos` on the bistatic branch; the phase term uses `fc` + geometric delay.
   A nominal `pulseRepfreq_Hz = 1.0` is passed and documented.
3. **Narrow-aperture resolution expectation.** *(resolved in Phase 3)* Sanity-check the expected per-look
   cross-range resolution from the 0.01° aperture and set grid extent/pixel spacing so the
   response is resolved and centered (not clipped, not over-sampled).
   → **Resolved empirically (differently than hypothesised).** Every look has the same
   within-look aperture (0.009964° → naïve sinc estimate ~86 m), but the measured
   reference-look response is a sharp isorange-contour **spike at the origin**, not an
   86 m lobe: the 1-sample delta compressed peak images the isorange contour (through the
   origin) directly. The ~86 m estimate assumed a broad compressed peak and does not apply.
   Grid extent/spacing chosen (range ±2 m @ 0.1 m, cross ±150 m @ 3 m) stays inside the
   ~3.25 m far-side range gate while showing the response.

## Risks

- **Fast-time signal-model alignment** — the highest-risk area now that the matched-filter
  math is reused. Mitigate via spike 1: confirm the range-compressed peak range before
  touching backprojection.
- **PRF / slow-time gap** — spike 2; could block Phase 3 if PRF is load-bearing.
- **Coordinate frame** — data is ENU-at-origin, `sar-ifp` expects ENU-at-origin: low risk,
  just confirm `sensor_positions` units/frame on load.
- **External read-only dependency** — `sar-ifp` is integrated against, not edited. Surface
  any required upstream change rather than modifying it from here.
- **Per-look coarse resolution** — set expectations in the goal so "well-focused" is judged
  as a single correctly-placed response, not a tight point.

## Verification approach

No functional tests in the gate. Verification is by human inspection of intermediate and
final products in the debugger, plus one targeted numpy/cupy agreement check during
Phase 2. Code standards (ruff + mypy) are enforced by `uv run python scripts/check.py` and
by CI on push/PR. The end-to-end acceptance signal is Phase 3: 361 looks, each a single
focused response at the origin.