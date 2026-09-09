"""Continuous carrier-tracking arc analysis for DROL corrected RINEX files.

An arc is a run of valid carrier-phase observations for one PRN.  It breaks
when the epoch gap exceeds ``gap_factor`` times the nominal sampling interval.
This is a tracking-continuity metric; it does not replace a cycle-slip test.
"""
from __future__ import annotations

import argparse
import math
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


def rinex_carrier_indices(path: Path):
    systems, pending = {}, {}
    with path.open("r", encoding="ascii", errors="ignore") as stream:
        for line in stream:
            if "END OF HEADER" in line: break
            if "SYS / # / OBS TYPES" not in line: continue
            system = line[0]
            parts = line[:60].split()
            if len(parts) >= 2 and parts[0] == system and parts[1].isdigit():
                pending[system] = int(parts[1]); systems[system] = parts[2:]
            elif system in pending:
                systems[system].extend(parts)
    return {key: [i for i, obs in enumerate(values[:pending[key]]) if obs.startswith("L")] for key, values in systems.items()}


def read_tracking(path: Path):
    carrier = rinex_carrier_indices(path)
    rows, time, header = [], None, True
    with path.open("r", encoding="ascii", errors="ignore") as stream:
        for line in stream:
            if header:
                if "END OF HEADER" in line: header = False
                continue
            if line.startswith(">"):
                fields = line[1:].split(); second = float(fields[5])
                time = datetime(*map(int, fields[:5]), int(second), round((second % 1) * 1_000_000))
                continue
            if time is None or len(line) < 4 or line[0] not in carrier or not line[1:3].isdigit(): continue
            indices = carrier[line[0]]
            if any(line[3 + 16*i:3 + 16*i + 14].strip() for i in indices):
                rows.append({"time": time, "prn": line[:3], "system": "GPS" if line[0] == "G" else "BDS", "file": path.name})
    return rows


def build_arcs(records: pd.DataFrame, gap_factor: float):
    arcs = []
    for (solution, prn), group in records.groupby(["solution", "prn"], sort=False):
        group = group.drop_duplicates("time").sort_values("time").reset_index(drop=True)
        if group.empty: continue
        seconds = group.time.astype("int64").to_numpy() / 1e9
        diffs = np.diff(seconds)
        nominal = float(np.median(diffs[(diffs > 0) & (diffs < 120)])) if np.any((diffs > 0) & (diffs < 120)) else 10.0
        breaks = np.r_[True, diffs > nominal * gap_factor]
        starts = np.flatnonzero(breaks)
        ends = np.r_[starts[1:] - 1, len(group) - 1]
        for start, end in zip(starts, ends):
            first, last = group.iloc[start], group.iloc[end]
            duration = (last.time - first.time).total_seconds() + nominal
            arcs.append({"solution": solution, "prn": prn, "system": first.system, "start": first.time,
                         "end": last.time, "epochs": end - start + 1, "duration_s": duration, "nominal_interval_s": nominal})
    return pd.DataFrame(arcs)


def plot_distribution(arcs: pd.DataFrame, output: Path, font_size: int, dpi: int):
    plt.rcParams.update({"font.family": "Arial", "font.size": font_size, "axes.labelsize": font_size + 2,
                         "axes.titlesize": font_size + 3, "legend.fontsize": font_size - 2, "savefig.dpi": dpi})
    order = ["antenna_1", "antenna_2", "dual_antenna_fused"]
    labels = {"antenna_1": "Ant1", "antenna_2": "Ant2", "dual_antenna_fused": "Fused"}
    colors = {"antenna_1": "#0072B2", "antenna_2": "#D55E00", "dual_antenna_fused": "#009E73"}
    # Compact paper-oriented bins. The first fine bin retains one-epoch arcs,
    # while the remaining bins make the long-arc contribution easy to read.
    edges = [-1e-6, 10 + 1e-6, 30 + 1e-6, 60 + 1e-6, 100 + 1e-6, 300 + 1e-6, 500 + 1e-6, np.inf]
    tick_labels = ["0-10", "10-30", "30-60", "60-100", "100-300", "300-500", ">500"]
    fig, ax = plt.subplots(figsize=(12.5, 6.6), layout="constrained")
    x = np.arange(len(tick_labels)); width = 0.25
    for index, solution in enumerate(order):
        position = x + (index - 1) * width
        gps, _ = np.histogram(arcs.loc[(arcs.solution.eq(solution)) & (arcs.system.eq("GPS")), "duration_s"], bins=edges)
        bds, _ = np.histogram(arcs.loc[(arcs.solution.eq(solution)) & (arcs.system.eq("BDS")), "duration_s"], bins=edges)
        # Colour encodes the antenna solution; fill style encodes constellation.
        ax.bar(position, gps, width=width, color=colors[solution], alpha=.82,
               edgecolor=colors[solution], linewidth=.9)
        ax.bar(position, bds, width=width, bottom=gps, color=colors[solution], alpha=.45,
               edgecolor=colors[solution], linewidth=.9, hatch="//")
    # Highlight the long-arc bins where antenna fusion is expected to be most useful.
    ymax = ax.get_ylim()[1]
    ax.add_patch(plt.Rectangle((3.5, 0), 3.0, ymax * .98, fill=False, edgecolor="#CC0000",
                               linewidth=1.8, linestyle=(0, (4, 3))))
    ax.set_xticks(x, tick_labels, rotation=25, ha="right")
    ax.set_xlabel("Continuous tracking arc duration (s)")
    ax.set_ylabel("Number of arcs")
    ax.set_title("Continuous carrier-tracking arc counts by duration")
    group_legend = ax.legend(handles=[Patch(facecolor=colors[key], edgecolor=colors[key], label=labels[key]) for key in order],
                             loc="upper left", title="Antenna solution")
    ax.add_artist(group_legend)
    ax.legend(handles=[Patch(facecolor="0.65", edgecolor="0.2", label="GPS (solid)"),
                       Patch(facecolor="0.65", edgecolor="0.2", hatch="//", label="BDS (hatched)")],
              loc="upper center", title="Constellation")
    ax.grid(True, axis="y", alpha=.25)
    fig.savefig(output / "continuous_arc_duration_counts.png", bbox_inches="tight"); plt.close(fig)


def plot_timeline(arcs: pd.DataFrame, day: str, output: Path, font_size: int, dpi: int):
    start = pd.Timestamp(day); end = start + pd.Timedelta(days=1)
    selected = arcs[(arcs.end >= start) & (arcs.start < end)].copy()
    if selected.empty: return
    plt.rcParams.update({"font.family": "Arial", "font.size": font_size, "savefig.dpi": dpi})
    order = ["antenna_1", "antenna_2", "dual_antenna_fused"]
    labels = {"antenna_1": "Ant1", "antenna_2": "Ant2", "dual_antenna_fused": "Fused"}
    colors = {"GPS": "#0072B2", "BDS": "#E69F00"}
    fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True, layout="constrained")
    for axis, solution in zip(axes, order):
        data = selected[selected.solution.eq(solution)].copy()
        prns = sorted(data.prn.unique())
        for y, prn in enumerate(prns):
            for _, arc in data[data.prn.eq(prn)].iterrows():
                left, right = max(pd.Timestamp(arc.start), start), min(pd.Timestamp(arc.end), end)
                axis.plot([left, right], [y, y], lw=4, solid_capstyle="butt", color=colors[arc.system])
        axis.set_yticks(range(len(prns))); axis.set_yticklabels(prns, fontsize=max(7, font_size - 5))
        axis.set_ylabel(labels[solution]); axis.grid(True, axis="x", alpha=.3)
    axes[0].set_title(f"Continuous carrier-tracking arcs on {day} (GPS time)")
    axes[-1].set_xlabel("GPS time")
    fig.savefig(output / f"continuous_arc_timeline_{day}.png", bbox_inches="tight"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="DROL continuous carrier-tracking arc analysis")
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\csu\dataset\DROL-GNSS"))
    parser.add_argument("--output", type=Path, default=Path(r"D:\csu\MyTools\analysis_output\DROL_batch"))
    parser.add_argument("--focus-day", default="20241227")
    parser.add_argument("--gap-factor", type=float, default=1.5)
    parser.add_argument("--font-size", type=int, default=15)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    inputs = {"antenna_1": sorted((args.data_root / "Obs_dualfreq").glob("DL0100*.rnx")),
              "antenna_2": sorted((args.data_root / "Obs_dualfreq").glob("DL0200*.rnx")),
              "dual_antenna_fused": sorted((args.data_root / "Obs_ant3").glob("DROL_*_AfterPCO_Merge2Ant.rnx"))}
    rows = []
    for solution, files in inputs.items():
        for path in files:
            rows.extend({"solution": solution, **row} for row in read_tracking(path))
    arcs = build_arcs(pd.DataFrame(rows), args.gap_factor)
    arcs.to_csv(args.output / "continuous_tracking_arcs.csv", index=False)
    stats = arcs.groupby(["solution", "system"], as_index=False).agg(arcs=("duration_s", "size"), mean_min=("duration_s", lambda x: x.mean()/60),
        median_min=("duration_s", lambda x: x.median()/60), p95_min=("duration_s", lambda x: x.quantile(.95)/60),
        over10min_pct=("duration_s", lambda x: 100*(x >= 600).mean()))
    stats.to_csv(args.output / "continuous_arc_statistics.csv", index=False, float_format="%.6f")
    plot_distribution(arcs, args.output, args.font_size, args.dpi)
    plot_timeline(arcs, args.focus_day, args.output, args.font_size, args.dpi)
    print(stats.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__": main()
