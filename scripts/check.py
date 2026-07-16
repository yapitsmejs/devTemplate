"""Project gate: lint + format check + type check.

Run before pushing or opening a PR::

    uv run python scripts/check.py

This project is a temporary, single-user toolbox for validating a radar
simulator, so the gate enforces only code standards (ruff check, ruff format
``--check``, mypy). Functional tests and branch coverage are intentionally not
part of the gate -- see ``README.md`` and ``CONVENTIONS.md``.
"""

from __future__ import annotations

import subprocess
import sys

# Lint/type-check target is the package import name. Keep in sync with
# pyproject.toml `packages = [...]` and `[tool.mypy] files`.
PACKAGE = "simValidation"


def _run(label: str, cmd: list[str]) -> bool:
    print(f"\n=== {label} ===\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd)  # noqa: S603 -- intentional, controlled args
    ok = result.returncode == 0
    print(f"--- {label}: {'PASS' if ok else 'FAIL'} ---")
    return ok


def main() -> int:
    py = sys.executable
    steps = [
        ("ruff check", [py, "-m", "ruff", "check", "."]),
        ("ruff format --check", [py, "-m", "ruff", "format", "--check", "."]),
        ("mypy", [py, "-m", "mypy", f"src/{PACKAGE}"]),
    ]
    for label, cmd in steps:
        if not _run(label, cmd):
            print("\n=== GATE: FAIL (stopped at " + label + ") ===")
            return 1
    print("\n=== GATE: PASS ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
