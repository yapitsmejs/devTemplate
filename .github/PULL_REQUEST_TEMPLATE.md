## Summary

<!-- What does this PR change, and why? -->

## Gate

- [ ] `uv run python scripts/check.py` passes
      (ruff check + ruff format --check + mypy)

## Checklist

- [ ] Public functions have type hints + Google-style docstrings
- [ ] `ruff format` applied; no new lint findings
- [ ] Naming follows `camelCase` + `_aspect` suffix (see `CONVENTIONS.md`)
- [ ] No changes made outside this repository (e.g. `sar-ifp`) unless explicitly
      told to