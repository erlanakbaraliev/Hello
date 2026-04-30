"""Ensure ``thesis/`` is on ``sys.path`` before ``import app`` / ``import database``.

Streamlit executes ``app/Home.py`` with the script directory on ``sys.path``, which is
``.../thesis/app`` — not the project root — so the ``app`` package is not importable
unless the parent directory (thesis) is prepended. Loaded via ``importlib`` from entry
scripts so this file never needs to be imported as ``app.invoke_bootstrap``.
"""

from __future__ import annotations

import sys
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    if (_p / "database.py").is_file() and (_p / "app").is_dir():
        _s = str(_p)
        if _s not in sys.path:
            sys.path.insert(0, _s)
        break
