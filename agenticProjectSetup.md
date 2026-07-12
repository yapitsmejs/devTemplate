# Agentic project setup

> **This file is a one-shot setup guide. Delete it (and `src/<pkg>/example.py`
> + `tests/test_example.py`) once the project is set up.** It exists only so an
> agent can bootstrap a new project from this template; it is not part of the
> shipped project and carries no lasting meaning.

An agent (any framework — Claude, Cursor, Codex, or other) reads this file and
follows it in order. Everything here is plain prose plus shell commands, so no
framework-specific mechanism is required.

## 1. Intake — ask the user these questions

Do **not** start renaming files until the user has answered. Ask them as a
conversation, adapting wording and skipping ones the user clearly doesn't care
about. The answers populate `README.md`, `CONVENTIONS.md`-adjacent context, and
`plans/developmentPlan.md`, so they need to be holistic enough to lay out a
development plan, not just label the project.

**Identity (drives the mechanical rename — see step 2):**
- Import name (camelCase, e.g. `myProject`) and dist name (kebab, e.g.
  `my-project`). Derive the dist name from the import name if the user doesn't
  give one.
- One-line description (fills `pyproject.toml` `description` and
  `src/<pkg>/__init__.py` docstring).
- Does it ship a CLI? If so, the entry command and a one-line invocation
  example.

**Purpose & audience (drives README intro + dev-plan rationale):**
- What problem does this project solve, and why now?
- Who uses it — end users, other devs as a library, CI/agents? Primary audience?
- What does "done / useful" look like at a first usable version?

**Scope & boundaries (drives the dev-plan phases):**
- In scope for v1 vs. deferred to later?
- Explicit **non-goals** — things people might assume it does but it won't.
- Hard success criteria or acceptance signals?

**Usage & integration (drives README usage + architecture):**
- How is it invoked/used? Inputs, outputs, expected shapes.
- Integrations — other systems, file formats, protocols, services?
- Runtime/platform constraints (OS, Python version beyond 3.12, GPU, network).

**Architecture & dependencies (drives dev-plan component breakdown + step 3):**
- Foreseeable components/modules?
- Which dependencies do you already know you want? (PyPI names and any version
  constraints.)
- Any dependencies you want to **avoid** (license, size, or philosophical
  reasons)?
- Persistence, state, concurrency/async, streaming, large data?

The agent will also **infer and suggest** further dependencies from your other
answers (e.g. a signal-processing project → `numpy`/`scipy`; a web API → an
ASGI server + a validation lib; a CLI → `typer`/`rich`; tests needing
parametrized data → `hypothesis`). It lists each suggestion with a one-line
rationale and waits for your confirmation before adding anything — see step 3.

**Non-functional (drives dev-plan risk/criteria sections):**
- Performance, scale, latency, or memory targets worth pinning now?
- Reliability, security, privacy, observability requirements?
- Anything that must be measurable (so the gate/tests can eventually enforce it)?

**Conventions & agent guidance:**
- Domain-specific naming, units, or rules beyond the repo's camelCase convention
  in `CONVENTIONS.md`?
- Standards or external specs the code must conform to?
- Anything an agent working in this repo should always know or always avoid?

**Development-plan inputs (drives `plans/developmentPlan.md`):**
- Phases or milestones you already have in mind, in rough order?
- Risky/unknown parts that need a spike or prototype first?
- Definition-of-done for the first milestone?
- Testing/verification approach beyond the built-in gate (property tests,
  integration, benchmarks)?

## 2. Mechanical rename — exact substitutions

Perform these precisely; they are deterministic string operations.

Let `importName` = the camelCase import name and `distName` = the kebab dist
name. Two tokens to replace everywhere: `myProject` → `importName` and
`my-project` → `distName`.

> The **root folder name is not your concern** — the user renames it manually
> before invoking you, and no file in this template references it. Do not attempt
> to rename the directory you are running inside; only rename paths *within* the
> project.

1. Rename the directory: `src/myProject/` → `src/<importName>/`.
2. In `pyproject.toml`:
   - `name = "my-project"` → `name = "<distName>"`.
   - `description = "<one-line project description>"` → the one-line description.
   - `packages = ["src/myProject"]` → `["src/<importName>"]` (under
     `[tool.hatch.build.targets.wheel]`).
   - `files = ["src/myProject"]` → `["src/<importName>"]` (under `[tool.mypy]`).
   - If a CLI was requested, uncomment the `[project.scripts]` block and set the
     entry to `<distName> = "<importName>.cli:main"` (adjust the target as the
     user specified).
3. `src/<importName>/__init__.py`: replace the docstring
   `"""<one-line project description>."""` with the one-line description.
4. Find-replace `myProject` → `importName` and `my-project` → `distName` across
   exactly these files (`.github/` and `.githooks/` contain no package name and
   need no changes):
   - `pyproject.toml`
   - `tests/test_smoke.py`
   - `tests/test_example.py` (deleted in step 4 anyway, but rename first if any
     import remains relevant)
   - `scripts/check.py` (both the `PACKAGE = "myProject"` constant and the
     docstring mention)
   - `CONVENTIONS.md`
   - `README.md` (also rewritten in step 3, so this is a safety net)

## 3. Dependencies — infer, propose, add

From the intake answers (purpose, use-case, architecture, non-functional),
infer the runtime dependencies the project likely needs. Start with the
dependencies the user already named, then propose additional ones your
inference suggests — each as `name>=lowerbound` with a one-line rationale
tying it to a specific answer (not generic "might be useful"). Keep
suggestions minimal and justified; a project that needs nothing should get
nothing.

Do **not** add anything silently. Present the full proposed set and get the
user's confirmation before editing files.

Add confirmed dependencies to `[project] dependencies` in `pyproject.toml`
(currently `dependencies = []`), one per line, e.g. `numpy>=1.26`. Keep dev
tooling (`pytest`, `ruff`, `mypy`) in `[project.optional-dependencies].dev` —
do not move it into core deps. Where the user gave no version constraint, pick a
conservative lower bound and note it. `uv sync --extra dev` in step 7 resolves
everything and updates `uv.lock`.

## 4. Prose — write the project's voice

From the intake answers, write/rewrite these. Keep `CONVENTIONS.md` as the
single source of truth for naming/code-style/TDD — do not duplicate it; the new
files point at it.

- **`README.md`** — rewrite fully from the answers: title (the project's name),
  one-line description, a Purpose section, an Install section (keep the existing
  `uv sync --extra dev` block), a Usage section with a runnable example, and a
  Development section (keep the existing `uv run ...` command block and the
  Automation subsection). Drop the "Starting a new project from this template"
  section and this setup prompt — the derived project is not a template.
- **`plans/developmentPlan.md`** — synthesize a phased roadmap from the
  scope/milestones/risks/DoD answers: milestones each with a definition-of-done,
  a component breakdown, dependencies between phases, flagged risks/spikes, and
  how each phase will be verified against the gate. This is the artifact the
  holistic intake exists to support.

## 5. Cleanup — delete template residue

- Delete `src/<importName>/example.py` and `tests/test_example.py`.
- Delete this file (`agenticProjectSetup.md`).
- Remove the "Starting a new project from this template" section from
  `README.md` (covered by the rewrite in step 3).

The 90% branch-coverage gate stays green after deleting the example pair:
`src/<importName>/__init__.py` becomes the only package module and is fully
covered by `tests/test_smoke.py` (it imports the package, hitting `__version__`
and `__all__`), with zero branches → 100% branch coverage ≥ 90. Verify this in
step 7 rather than assuming it.

## 6. Placeholder sweep — verify nothing templated remains

Search the repo (excluding `.venv/`, `.mypy_cache/`, `.ruff_cache/`,
`.pytest_cache/`, `uv.lock`) for residual template markers and fix any
stragglers:

```bash
grep -rn -E '<[a-zA-Z]|myProject|my-project|my_project' \
  --exclude-dir=.venv --exclude-dir=.mypy_cache --exclude-dir=.ruff_cache \
  --exclude-dir=.pytest_cache --exclude=uv.lock .
```

Expected: zero hits outside this file (which is being deleted). Any hit means a
substitution was missed — fix it.

## 7. Gate — run it and report

```bash
uv sync --extra dev
uv run python scripts/check.py
```

Report what you did (rename, files written/deleted) and paste the gate output.
If the gate fails, fix it before handing back — a red skeleton is not a
finished setup.