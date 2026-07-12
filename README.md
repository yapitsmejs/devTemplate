# myProject

<One-line project description.>

## Install

This project is managed with [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev          # create .venv and install dev dependencies (pytest, ruff)
```

## Usage

<Describe how to run the project.>

```python
import myProject

print(myProject.__version__)
```

## Development

```bash
uv run pytest                 # run tests (dev loop, no coverage gate)
uv run ruff check .           # lint
uv run ruff format .          # format
uv run mypy src/myProject     # type check
uv run python scripts/check.py   # GATE: lint + format + types + tests + branch coverage ≥ 90%
```

Run `scripts/check.py` before pushing or opening a PR — it fails if ruff is
unhappy, formatting would change, **mypy** finds a type error, or branch coverage
on the package drops below 90%. See [`CONVENTIONS.md`](CONVENTIONS.md) for the
full naming, code-style, and test-driven conventions.

### Automation

These run the same gate as you do locally, so a red PR cannot merge:

- **CI** (`.github/workflows/ci.yml`) runs `scripts/check.py` on every push and PR.
- **Git hooks** (`.githooks/`) are opt-in per clone — fast `ruff` checks on
  commit, the full gate on push. One-time setup:
  ```bash
  git config core.hooksPath .githooks
  git update-index --chmod=+x .githooks/pre-commit .githooks/pre-push
  ```
  Use `git commit --no-verify` during active red-phase work; the pre-push hook
  and CI still enforce before anything lands.
- **Dependabot** (`.github/dependabot.yml`) opens weekly dependency PRs (uv +
  GitHub Actions) that must pass the gate like any other.
- **PR template** (`.github/PULL_REQUEST_TEMPLATE.md`) prompts contributors for
  the red-green workflow and the gate checklist.

## Starting a new project from this template

Copy this directory, rename the folder to your project's name, `git init` it,
and hand the following prompt to your agent (any framework — Claude, Cursor,
Codex, or other). Rename the **root folder yourself before invoking the
agent** — the agent renames everything inside the project (the package
directory and every file), but it cannot reliably rename the folder it is
running inside. The folder name is cosmetic here: no file references it, so it
does not affect the gate.

```
Set up a new project from this template. Read `agenticProjectSetup.md` and
follow every step in order: ask me the intake questions first, then do the
mechanical rename and find-replace, propose dependencies to add, rewrite this
README and create `plans/developmentPlan.md`, delete the example files and
`agenticProjectSetup.md`, run a placeholder sweep, and run the gate
(`uv run python scripts/check.py`). Report what you did and paste the gate
output.
```

The agent reads `agenticProjectSetup.md` for the full runbook (intake question
set, exact substitutions, prose to write, cleanup, and the gate). That file is
a throwaway setup guide — it is deleted as part of setup and is not part of the
shipped project.