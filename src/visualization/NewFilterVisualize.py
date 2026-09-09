"""Backward-compatible import path for filter diagnostic figures."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plot.filter import *  # noqa: F401,F403
