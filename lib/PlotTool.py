"""Backward-compatible import path for the consolidated plotting package."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plot.legacy import *  # noqa: F401,F403
