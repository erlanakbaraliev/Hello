"""Single-command test runner for the Air Quality project.

Usage:
    python run_tests.py           # run the full pytest suite with coverage
    python run_tests.py -k name   # forward extra args to pytest
    python run_tests.py --no-cov  # run without coverage reporting

Coverage settings (source paths, omits, fail-under threshold) live in
``pyproject.toml`` under ``[tool.pytest.ini_options]`` and ``[tool.coverage.*]``.
A browsable HTML report is written to ``coverage_html/`` after each run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TESTS_DIR = ROOT / "src" / "tests"


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    cmd: list[str] = [sys.executable, "-m", "pytest", str(TESTS_DIR)]

    if "--no-cov" in args:
        args.remove("--no-cov")
        cmd += [
            "-o",
            "addopts=",  # wipe the cov-related defaults from pyproject
        ]

    cmd.extend(args)
    print("$", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
