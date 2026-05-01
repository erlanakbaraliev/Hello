"""Launcher for the Air Quality Streamlit app.

Run with:
    python app.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    home = Path(__file__).resolve().parent / "src" / "front_end" / "Home.py"
    if not home.is_file():
        print(f"Cannot find Streamlit entry point at {home}", file=sys.stderr)
        return 1
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(home)])


if __name__ == "__main__":
    raise SystemExit(main())
