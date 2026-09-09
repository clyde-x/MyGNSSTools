"""Reusable, file-oriented figures for DROL analysis workflows.

Functions accept already computed tables and write figures to ``output``;
they do not parse source data or mutate analysis tables.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def configure(font_size: int = 15, dpi: int = 300) -> None:
    """Set shared, paper-oriented Matplotlib defaults for DROL figures."""
    plt.rcParams.update({"font.family": "Arial", "font.size": font_size,
                         "axes.labelsize": font_size + 1, "axes.titlesize": font_size + 2,
                         "legend.fontsize": max(1, font_size - 2), "savefig.dpi": dpi})


def plot_observation_availability(epoch: pd.DataFrame, output: Path, focus_day: str,
                                  styles: dict[str, tuple[str, str]] | None = None,
                                  filename_prefix: str = "observation") -> Path | None:
    """Plot total/GPS/BDS epoch availability for the selected day."""
    focus = epoch[epoch.day.eq(focus_day)]
    if focus.empty:
        return None
    styles = styles or {"antenna_1": ("Ant1", "#0072B2"), "antenna_2": ("Ant2", "#D55E00"),
                        "dual_antenna_fused": ("Fused", "#009E73")}
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, layout="constrained")
    for key, (label, color) in styles.items():
        group = focus[focus.solution.eq(key)]
        for axis, column in zip(axes, ("total", "gps", "bds")):
            if not group.empty:
                axis.plot(group.time, group[column], lw=.8, label=label, color=color)
    for axis, ylabel, column in zip(axes, ("Observed satellites", "GPS satellites", "BDS satellites"),
                                    ("total", "gps", "bds")):
        axis.set_ylabel(ylabel); axis.grid(True, alpha=.3); axis.axhline(4, color="0.5", lw=1.2, ls="--", alpha=.7)
        means = focus.groupby("solution")[column].mean()
        axis.text(.015, .96, "\n".join(f"{styles[key][0]}: {means[key]:.2f}" for key in styles if key in means),
                  transform=axis.transAxes, va="top", fontsize=plt.rcParams["font.size"] - 1,
                  bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.5", alpha=.88))
    axes[0].set_title(f"DROL observation availability on {focus_day}")
    axes[0].legend(ncol=3, loc="upper right"); axes[-1].set_xlabel("UTC time")
    path = output / f"{filename_prefix}_timeseries_{focus_day}.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def plot_force_rtn(frame: pd.DataFrame, groups: dict[str, tuple[str, str, str]], output: Path,
                   font_size: int, dpi: int, max_points: int = 12000) -> tuple[Path, Path]:
    """Write component and magnitude figures for inferred RTN forces."""
    configure(font_size, dpi)
    shown = frame.iloc[::max(1, int(np.ceil(len(frame) / max_points)))]
    components = ("R", "T", "N"); colors = ("#0072B2", "#D55E00", "#009E73")
    fig, axes = plt.subplots(len(groups), 1, figsize=(14, 13), sharex=True, layout="constrained")
    for axis, (name, columns) in zip(np.atleast_1d(axes), groups.items()):
        for component, column, color in zip(components, columns, colors):
            axis.plot(shown.utc, shown[column], lw=.65, label=component, color=color)
        axis.set_ylabel("Acceleration (m/s²)"); axis.set_title(name); axis.grid(True, alpha=.28); axis.legend(ncol=3, loc="upper left")
    axes[-1].set_xlabel("UTC time (from MJD)"); axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    component_path = output / "force_rtn_inferred_components.png"; fig.savefig(component_path, bbox_inches="tight"); plt.close(fig)
    fig, axis = plt.subplots(figsize=(14, 6.2), layout="constrained")
    for (name, columns), color in zip(groups.items(), ("#0072B2", "#D55E00", "#009E73", "#CC79A7")):
        axis.semilogy(shown.utc, np.linalg.norm(shown[list(columns)].to_numpy(), axis=1), lw=.75, label=name, color=color)
    axis.set(title="Inferred force magnitudes in RTN frame (header absent)", ylabel="Acceleration magnitude (m/s²)", xlabel="UTC time (from MJD)")
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M")); axis.grid(True, which="both", alpha=.28); axis.legend(ncol=2)
    magnitude_path = output / "force_rtn_inferred_magnitudes.png"; fig.savefig(magnitude_path, bbox_inches="tight"); plt.close(fig)
    return component_path, magnitude_path


def plot_frequency_counts(frame: pd.DataFrame, system: str, output: Path, day: str,
                          font_size: int, dpi: int) -> Path | None:
    """Plot per-PRN valid code-observation counts for one constellation."""
    subset = frame[frame.system.eq(system)].copy()
    if subset.empty:
        return None
    configure(font_size, dpi)
    kinds = sorted(subset.observation_type.unique()); prns = sorted(subset.prn.unique(), key=lambda value: int(value[1:]))
    pivot = subset.pivot(index="prn", columns="observation_type", values="valid_observations").reindex(prns).fillna(0)
    fig, axis = plt.subplots(figsize=(13, 6.4), layout="constrained"); x = np.arange(len(prns)); width = min(.26, .78 / len(kinds))
    for index, kind in enumerate(kinds):
        axis.bar(x + (index - (len(kinds) - 1) / 2) * width, pivot[kind], width, label=kind,
                 color=("#E41A1C", "#0055CC", "#18A638", "#984EA3")[index % 4], edgecolor="none")
    name = {"G": "GPS", "C": "BDS"}[system]
    axis.set_xticks(x, [prn[1:] for prn in prns]); axis.set(xlabel=f"{name} PRN", ylabel="Number of valid observations", title=f"{name} L1B code-observation availability ({day})")
    axis.legend(title="Observation type", ncol=len(kinds), loc="upper center"); axis.grid(axis="y", alpha=.28); axis.set_axisbelow(True)
    path = output / f"l1b_{system.lower()}_frequency_counts_{day}.png"; fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path
