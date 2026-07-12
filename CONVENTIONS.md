# Conventions

This document is the single source of truth for naming and code style in this
project. It is intentionally tool-agnostic — it does not assume any particular
editor, linter UI, or AI assistant. `CLAUDE.md` (when present) points here.

## Naming conventions

This project uses **camelCase for the name and a snake_case underscore as a
separator for different aspects** of an identifier. The underscore-separated tail
carries a unit, coordinate frame, or other aspect; the camelCase part carries
the quantity's name.

- **Variables:** `camelCase` + `_aspect` suffix.
  - Units: `heightOfX_m`, `targetBeamAngle_rad`, `radarCenterFrequency_Hz`,
    `aperturePowerBound_dB`, `fastTimeAxis_s`, `wavePropogationSpeed_mPerSec`,
    `pulseRepFreq_Hz`.
  - Coordinate frames as a composite suffix: `apcStartPos_XYZ_m`,
    `apcVelVec_XYZ_mPerSec`, `sceneCenter_ENU_m`.
  - Unitless quantities are plain camelCase: `pixelWiseDelay`, `interpValues`,
    `phaseCorrection`.
- **Modules / files & the import package:** camelCase — `normalizeSignal.py`,
  `computeStats.py`, the package `myProject/`. Private modules lead with an
  underscore: `_chunking.py`, `_phaseCorrelationCore.py`.
- **Functions:** camelCase public — `formImage`, `computeFbr`, `getTxPos`.
  Private helpers are `_camelCase` — `_robustMode`, `_selfcheck`. Backend-dispatch
  compute cores use the `_<func>_core(..., xp)` pattern (one code path for
  numpy/cupy, no mirrored `_<func>_cupy` duplicates): `_formImage_core`,
  `_computeFbr_core`.
- **Classes:** PascalCase — `Transaction`, `Statement`, `Level`.
- **Constants:** UPPER_SNAKE. Private constants keep a single leading underscore
  — `_HAVE_CUPY_GPU`, `_STMT_DATE_RE`, `_SKIP`.

## Code rigor

- Start every module with `from __future__ import annotations` (deferred
  evaluation, enables PEP 604 on older targets).
- **Type hints are required** on public functions. Use PEP 604/585 style:
  `x | None`, `list[int]`, `dict[str, float]`, `tuple[float, float] | None`. Do
  not use `typing.Optional` / `typing.List`. Enforced by **mypy**
  (`uv run mypy src/myProject`, and via the gate) — the gate fails on an untyped
  public function.
- **Group imports** in order: future → stdlib → third-party → local, with a
  blank line between groups. Let `ruff` (rule `I`) own the ordering.
- **Google-style docstrings** on every public function:
  ```python
  def formImage(signal, rangeAxis_m):
      """Form a focused image.

      Args:
          signal: Range-compressed signal, shape (nRange, nPulse).
          rangeAxis_m: Output range axis, in meters.

      Returns:
          Focused image as a 2-D array.
      """
  ```
- Declare the public API with `__all__` in `__init__.py` and in tool modules
  that export helpers.
- **ruff** selects `E, F, W, I, UP, B` and ignores `E501`. `N` (pep8-naming) is
  deliberately excluded so camelCase identifiers are not flagged — do not add it.

## Testing / Test-driven development

This project is test-driven. Features land with tests, and a coverage gate
keeps it that way.

- **Test-first (red-green-refactor):** before writing or changing behavior,
  write or extend the relevant `tests/test_<module>.py` test and run it red
  (`uv run pytest tests/test_<module>.py`). Then write the minimal
  implementation to turn it green, then refactor.
- **Dev loop:** `uv run pytest` runs the full suite fast, with **no coverage
  enforcement** — use it during red/green work.
- **Gate:** `uv run python scripts/check.py` runs ruff check, ruff format
  `--check`, **mypy**, and pytest with branch coverage. Branch coverage on the
  package must stay **≥ 90%**. Run it (and it must pass) before pushing or opening
  a PR. CI (`.github/workflows/ci.yml`) runs the same gate on every push and PR,
  so a red PR cannot merge.
- **Test naming:** `test` + camelCase name, optionally a `_aspect` suffix —
  `testVersionIsSet`, `testClampValue_clampsBelowLow`. (Same camelCase rule as
  the rest of the codebase; `N`/pep8-naming stays excluded in ruff.)
- **Layout:** one `tests/test_<module>.py` per source module
  (`example.py` → `tests/test_example.py`). `test_smoke.py` is the lone
  cross-cutting smoke test.

## Commit convention

Use conventional commit types: `feat`, `fix`, `docs`, `style`, `refactor`,
`perf`, `test`, `build`, `ci`, `chore`, `revert`.