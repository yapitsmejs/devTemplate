# simValidation

Temporary tools to validate a radar simulator.

## Purpose

This is a small, single-user repository of throwaway tools for validating a
radar simulator. It is **not** a development project and **not** meant to ship:
after the simulator is validated, a more full-purpose interface will be built
separately for a team of developers to use. Everything here exists only to
answer "is the simulator producing correct SAR data?" through human inspection.

The first usable version is a pipeline that reads the raw data pulses (and
reference pulse) produced by the simulator, performs a matched filter, and
backprojects the result into a SAR image. The supplied data is a single trihedral
target imaged from 360 viewing angles, so "the simulator looks right" means a
well-focused point target in the reconstructed image.

## Install

This project is managed with [`uv`](https://docs.astral.sh/uv/). Python 3.12 is
required, and `cupy` is expected to be installable on this machine (CUDA wheels
are resolved by `uv`).

```bash
uv sync --extra dev          # create .venv and install runtime + dev dependencies
```

Runtime dependencies are `numpy`, `scipy`, `matplotlib`, `cupy` (GPU
acceleration), and `psutil` (read available RAM for automatic data chunking).

## Usage

There is no CLI. Tools are run as scripts, typically under a debugger so
intermediate variables (raw pulses, matched-filter output, the focused image)
can be inspected directly.

The intended data flow is:

1. **Read raw pulses** from the simulator output directory
   (`C:\Users\yJoonSio\OneDrive - DSO\trihedral`) — the phase-history pulses plus
   the reference pulse.
2. **Matched filter** the pulses against the reference.
3. **Backproject** the matched-filtered data into a SAR image, using the
   backprojection packages from the local `sar-ifp` repository
   (`C:\Users\yJoonSio\Desktop\ghRepos\sar-ifp`).

A representative script (planned shape — modules land per the development plan):

```python
from __future__ import annotations

import simValidation as sv

# 1. Read raw pulses + reference pulse from the simulator output directory.
pulses, referencePulse, metadata = sv.readPulses(
    r"C:\Users\yJoonSio\OneDrive - DSO\trihedral"
)

# 2. Matched filter (cupy-accelerated, chunked to fit VRAM/RAM).
mfSignal = sv.matchFilter(pulses, referencePulse)

# 3. Backproject into a SAR image via the local sar-ifp backprojection packages.
image = sv.backproject(mfSignal, metadata)

# Inspect `image` in the debugger -- expect a well-focused point target.
```

> Do **not** modify anything outside this repository unless explicitly told — in
> particular, treat `sar-ifp` as a read-only dependency to integrate against, not
> something to edit from here.

## Development

```bash
uv run ruff check .           # lint
uv run ruff format .          # format
uv run mypy src/simValidation # type check
uv run python scripts/check.py   # GATE: lint + format + types (code standards only)
```

Run `scripts/check.py` before pushing or opening a PR — it fails if ruff is
unhappy, formatting would change, or **mypy** finds a type error. There are
**no functional tests and no coverage gate** in this project: it is a temporary
validation toolbox, and results are judged by human inspection (e.g. a focused
point target) rather than by assertions. See
[`CONVENTIONS.md`](CONVENTIONS.md) for the full naming and code-style
conventions, and [`plans/developmentPlan.md`](plans/developmentPlan.md) for the
phased roadmap.

### Automation

These run the same gate as you do locally, so a red PR cannot merge:

- **CI** (`.github/workflows/ci.yml`) runs `scripts/check.py` on every push and PR.
- **Git hooks** (`.githooks/`) are opt-in per clone — fast `ruff` checks on
  commit, the full gate on push. One-time setup:
  ```bash
  git config core.hooksPath .githooks
  git update-index --chmod=+x .githooks/pre-commit .githooks/pre-push
  ```
  Use `git commit --no-verify` during active work; the pre-push hook and CI
  still enforce before anything lands.
- **Dependabot** (`.github/dependabot.yml`) opens weekly dependency PRs (uv +
  GitHub Actions) that must pass the gate like any other.
- **PR template** (`.github/PULL_REQUEST_TEMPLATE.md`) prompts contributors for
  the gate checklist.