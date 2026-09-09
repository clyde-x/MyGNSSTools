"""Unified public plotting interface for MyTools.

Use the domain modules rather than importing plotting helpers from analysis
scripts.  The former ``lib.PlotTool`` and ``src.visualization`` paths remain
available as compatibility shims.
"""

__all__ = ["drol", "filter", "legacy", "orbit", "residual", "satellite_count"]
