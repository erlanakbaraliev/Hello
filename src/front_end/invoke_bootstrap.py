"""Ensure ``src/`` is on ``sys.path`` before ``from front_end ...`` / ``from db ...``.

Streamlit executes ``src/front_end/Home.py`` (main) or a script under ``pages/`` with
the script directory on ``sys.path`` (``.../src/front_end`` or ``.../src/front_end/pages``),
not the project root or ``src/``. Without
adding ``src/`` to ``sys.path`` the ``front_end``, ``back_end`` and ``db``
packages would not be importable from inside the Streamlit pages.

Loaded via ``importlib`` from entry scripts so this file never needs to be
imported as ``front_end.invoke_bootstrap``.
"""

from __future__ import annotations

import sys
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    if (_p / "db").is_dir() and (_p / "front_end").is_dir():
        _s = str(_p)
        if _s not in sys.path:
            sys.path.insert(0, _s)
        break
