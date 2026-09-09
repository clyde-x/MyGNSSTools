"""Compatibility entry point; use ``drol_dop_comparison.py`` for all new work."""
from __future__ import annotations

import warnings

from drol_dop_comparison import main


if __name__ == "__main__":
    warnings.warn("drol_dop_analysis.py is retained for compatibility; use drol_dop_comparison.py.", DeprecationWarning)
    main()
