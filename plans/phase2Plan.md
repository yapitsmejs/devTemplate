# Phase 2 plan — matched filter + signal-model validation

> Implementation plan for Phase 2 of [`developmentPlan.md`](./developmentPlan.md).
> Throwaway toolbox; gate is code standards only (ruff + mypy). See
> [`CONVENTIONS.md`](../CONVENTIONS.md) for naming/style.

## Context

Phase 1 (data reader) is implemented and gate-clean. A prior manual attempt to
matched-filter the simulator's pulses against the provided reference pulse **failed to
compress the pulse**. Two read-only explorations (sar-ifp source + empirical diagnosis of
the actual data) pin down exactly why, and confirm the planned approach.

**sar-ifp signal model (source-confirmed):**
- `sarIfp.sarSim.applyMatchedFilter(s, h)` is exactly `ifft(fft(s, axis=0) · conj(fft(h)))`
  per pulse (per column, axis-0 = fast-time), **no zero-padding**, `h` conjugated in the
  frequency domain. 2-D `s` broadcasts one 1-D `h` across all columns (`sarSim.py:189-197`).
- The matched-filter reference is the **clean transmit replica** — `getBbChirp(tAxis, b, tp,
  fc)` with `t0 = tAxis[0]`, `at = 1` (`sarSim.py:373`); an up-chirp
  `exp(j·π·(b/tp)·(t−tp/2−t0)²)` on `[t0, t0+tp]`, rectangular window. Not a calibration echo.
- **Dechirp is declared but unimplemented everywhere** — `formImage`'s dechirp branch
  returns a zero grid (`backProjection.py:110,140`; `tests/test_backProjection.py:69-84`).
- Fast-time axis is **absolute seconds-since-transmit**:
  `fastTimeAxis_s = arange(N)/fs + gateStart_s`; the compressed peak's `tAxis_s[k]` equals
  the absolute two-way delay → `range = c·tAxis_s[k]/2` (`getRxSamplingWindow`,
  `sarSim.py:469`).

**Empirical diagnosis (look 0, pulse 0, read-only numpy):**
- `rx_sig` pulses are **raw LFM** (broadband down-chirp; 90% energy across 0.82 of Nyquist;
  STFT instantaneous-frequency ramps at ~−2.04e14 Hz/s). Plain `fft(s)` does not compress
  (rel width 0.37) → **not** already-dechirped → standard matched filter is correct.
- `lfm_ref` is a valid clean up-chirp replica: normalized cross-correlation vs ideal
  `exp(+j·π·K·t²)` = **0.9995**. MF with `h=lfm_ref` gives **width 2 samples (rel 0.0020)**,
  identical to MF with the ideal chirp. → Use `lfm_ref` directly; do not synthesize a reference.
- **Root cause of the prior failure:** the manual code used `correlate(s, conj(h[::-1]))`,
  which **double-reverses** (`np.correlate` already reverses its 2nd argument) → effectively
  convolves → width **19**, peak 22× lower. Correct forms: `correlate(s, conj(h), 'same')`
  or the frequency-domain `ifft(fft(s)·conj(fft(h)))` (both width 2). Reusing
  `applyMatchedFilter` sidesteps this entirely.
- Peak idx **987**, **bit-identical** across pulses 0…1199 → fixed point target at origin.
- Spike 1 resolution: with the freq-MF peak at idx 987, `gateStart = 2·slantRange_m/c −
  idx/fs ≈ 113.9175 µs` makes `fastTimeAxis_s[987] = 2R/c` (target 17,322.385 m), matching
  `formImage`'s geometric-delay interpolation. Target sits near the far edge of the ~250 m
  gate window (far-side budget ~3.4 m) — a Phase 3 grid constraint, not a Phase 2 blocker.

## Deliverables

1. **`src/simValidation/matchFilter.py`** (new) — reuses `applyMatchedFilter`, owns the
   per-look iteration, cupy opt-in, and the fast-time-axis / signal-model alignment:
   - `matchFilter(pulses, referencePulse, metadata, useGpu=None) -> np.ndarray` —
     range-compress. Accepts the mmaped cube or a slice `(nLooks, 1200, 1001)`; per look,
     transpose to `(1001, 1200)` (fast-time × pulses) and call
     `sarIfp.sarSim.applyMatchedFilter`; returns `(nLooks, 1001, 1200)` complex128.
     `useGpu=None` auto-uses cupy when `_HAVE_CUPY_GPU` (guarded import mirroring sar-ifp),
     else numpy; an explicit `useGpu=` overrides. Memory note: a full 361-look result is
     ~6.9 GB — Phase 2 validation passes a small slice; Phase 3 will likely stream
     look-by-look into `formImage` rather than hold the whole compressed cube.
   - `estimateGateStart_s(compressedLook, metadata) -> float` —
     `idxPeak = argmax(|compressedLook|)`; `gateStart_s = 2·metadata["slantRange_m"]/c −
     idxPeak/metadata["rxSamplingFreq_Hz"]`.
   - `buildFastTimeAxis(metadata, gateStart_s) -> np.ndarray` —
     `np.arange(metadata["nRangeSamples"]) / metadata["rxSamplingFreq_Hz"] + gateStart_s`,
     float64, tx-referenced (matches sar-ifp convention).
   - `__all__`, Google docstrings, full type hints; `from __future__ import annotations`.
2. **`src/simValidation/__init__.py`** — add `matchFilter`, `buildFastTimeAxis`,
   `estimateGateStart_s` to `__all__` and re-export.
3. **`scripts/inspectMatchFilter.py`** (new) — the validation harness (see below).
4. **`plans/developmentPlan.md`** — append a short "Phase 2 — resolved" note recording the
   root cause (double-reversal), the `lfm_ref`-is-valid finding, and the `gateStart`
   calibration.
5. **`pyproject.toml`** — add `sar-ifp` as a local-path (editable) dependency via
   `[tool.uv.sources]` so `import sarIfp` works under `uv sync` without touching the
   read-only `sar-ifp` repo.

## Validation — `scripts/inspectMatchFilter.py` (acceptance criteria)

Run under a debugger or headless; prints a structured report and exits 0 only if all pass:

1. **Compression works:** compress look 0 (numpy); compute −3 dB width of the main peak;
   assert **rel width < 0.05** (expect ~0.002). Report peak value, idx (expect 987), width.
2. **Root-cause demonstration:** run the buggy `np.correlate(s, np.conj(h[::-1]), "same")`
   and report its width (expect ~19) beside the correct freq-MF width (~2) — documents
   why the manual attempt failed and that the new path fixes it.
3. **numpy vs cupy agreement (Phase 2 DoD):** compress look 0 on both backends; assert
   `max|abs(np_out − cp.asnumpy(cp_out))` relative < 1e-9. Skip with a printed SKIP if no GPU.
4. **Peak stability:** compress pulses 0, 100, 200, …, 1199 of look 0; assert all peak idx
   equal (expect 987) and peak values bit-identical.
5. **Peak → range (spike 1):** `estimateGateStart_s` → `buildFastTimeAxis` → assert
   `fastTimeAxis_s[idxPeak] · c / 2 ≈ metadata["slantRange_m"]` (exact by construction);
   report `gateStart_s` (expect ~1.139e-4 s) and the within-gate range (0 < r < 250 m).
6. **Reference validity:** `|xcorr(lfm_ref, ideal up-chirp)| > 0.99` (expect 0.9995), where
   the ideal chirp is built from `metadata` (`exp(j·π·K·t²)`, `K = bandwidth/duration`,
   `t = arange(N)/fs`).

## Spikes carried to Phase 3 (not blocking Phase 2)

- **Circular-wrap / focus alignment:** confirm `formImage` focuses the target with
  `gateStart` derived from the freq-MF peak (idx 987). If focus fails, re-anchor using the
  unwrapped (dechirp) peak idx 14 instead and re-test. Resolve empirically in Phase 3.
- **Grid extent vs range-gate window:** the ~250 m gate window spans slant range
  [17075.8, 17325.8] m with the target near the far edge; the ground grid must keep all
  pixel slant ranges inside this window to avoid interp extrapolation artifacts (far-side
  budget ~3.4 m). Sized in Phase 3's `formGrid`.
- **PRF usage:** read `backProjection._formImage_core` before backprojection to confirm
  whether `pulseRepfreq_Hz` actually enters the spotlight phase term (the per-pulse term is
  `exp(2j·π·fc·delay)`, which uses `fc` + geometric delay, not PRF). If vestigial, pass a
  nominal PRF and document; if load-bearing, derive from angular spacing + an assumed
  platform velocity and flag the assumption.

## Verification

- `uv run python scripts/check.py` → GATE PASS (ruff check + format --check + mypy).
- `uv run python scripts/inspectMatchFilter.py` → all six acceptance criteria pass (exit 0).
- `uv run python scripts/inspectReader.py` → still passes (no Phase 1 regression).

## Execution order

1. Wire `sar-ifp` into `pyproject.toml`; `uv sync`; confirm `import sarIfp`.
2. Write `matchFilter.py` + update `__init__.py`.
3. `ruff check --fix . && ruff format .` then `scripts/check.py` until gate is green.
4. Write `scripts/inspectMatchFilter.py`; run it; iterate until all criteria pass.
5. Append the Phase 2 "resolved" note to `plans/developmentPlan.md`; re-run the gate.