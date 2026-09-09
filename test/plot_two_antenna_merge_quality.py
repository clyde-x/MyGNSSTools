"""Create availability and continuous-arc figures for the current two-antenna merge.

Run with the GNSS Conda environment:
    C:\\Users\\Eren\\anaconda3\\envs\\GNSS\\python.exe plot_two_antenna_merge_quality.py
"""
from __future__ import annotations

import ast
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(r"D:\csu\MyTools")
TARGET = ROOT / "lib" / "Merge_TwoAnts_Rinex.py"
PRIMARY = Path(r"D:\csu\dataset\DROL-GNSS\Obs_pco\DL0200LEO_20243600_PCO.rnx")
SECONDARY = Path(r"D:\csu\dataset\DROL-GNSS\Obs_pco\DL0100LEO_20243600_PCO.rnx")
OUT = ROOT / "test" / "merge_quality_figures"
STYLES = {"primary": ("Primary (DL0200)", "#0072B2"),
          "secondary": ("Secondary (DL0100)", "#D55E00"),
          "fused": ("Fused (current algorithm)", "#009E73")}


def load_merge():
    """Load the production merge functions while avoiding its GUI-only imports."""
    tree = ast.parse(TARGET.read_text(encoding="utf-8"))
    names = {"get_satellite_segments", "merge_rinex_data_with_strategy1"}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = {"defaultdict": defaultdict, "timedelta": timedelta}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(TARGET), "exec"), scope)
    return scope["merge_rinex_data_with_strategy1"]


def read_rinex(path: Path):
    data, current, header = {}, None, True
    for line in path.open("r", encoding="ascii", errors="ignore"):
        if header:
            header = "END OF HEADER" not in line
            continue
        if line.startswith(">"):
            parts = line[1:].split(); second = float(parts[5])
            current = datetime(*map(int, parts[:5]), int(second), round((second % 1) * 1_000_000))
            data[current] = {"time": current, "flag": "0", "num_satellites": 0, "observations": {}}
        elif current is not None and len(line) >= 3 and line[0] in {"G", "C"} and line[1:3].isdigit():
            data[current]["observations"][line[:3]] = line.rstrip()
    for item in data.values(): item["num_satellites"] = len(item["observations"])
    return data


def count_records(solutions):
    rows = []
    for solution, data in solutions.items():
        for time, value in data.items():
            sats = value["observations"]
            rows.append({"solution": solution, "time": time,
                         "GPS": sum(s[0] == "G" for s in sats), "BDS": sum(s[0] == "C" for s in sats),
                         "total": len(sats)})
    return pd.DataFrame(rows).sort_values(["solution", "time"])


def continuous_arcs(data):
    appearances = defaultdict(list)
    for time, value in data.items():
        for sat in value["observations"]: appearances[sat].append(time)
    all_times = sorted(data)
    cadence = min((b - a).total_seconds() for a, b in zip(all_times, all_times[1:]) if b > a)
    rows = []
    for sat, times in appearances.items():
        times = sorted(times); start = previous = times[0]
        for time in times[1:]:
            if (time - previous).total_seconds() > 1.5 * cadence:
                rows.append({"satellite": sat, "system": sat[0], "start": start, "end": previous,
                             "duration_s": (previous - start).total_seconds()})
                start = time
            previous = time
        rows.append({"satellite": sat, "system": sat[0], "start": start, "end": previous,
                     "duration_s": (previous - start).total_seconds()})
    return pd.DataFrame(rows), cadence


def plot_counts(epoch: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True, layout="constrained")
    for solution, (label, color) in STYLES.items():
        group = epoch[epoch.solution.eq(solution)]
        for axis, column in zip(axes, ["total", "GPS", "BDS"]):
            axis.plot(group.time, group[column], lw=.65, color=color, label=label)
    for axis, label in zip(axes, ["Observed satellites", "GPS satellites", "BDS satellites"]):
        axis.set_ylabel(label); axis.grid(alpha=.28)
    axes[0].set_title("Per-epoch satellite availability: primary, secondary, and current fusion")
    axes[0].legend(ncol=3, loc="upper right"); axes[-1].set_xlabel("UTC time")
    fig.savefig(OUT / "satellite_count_timeseries.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_arc_statistics(arcs: pd.DataFrame):
    summary = arcs.groupby("solution").agg(arcs=("satellite", "size"), total_hours=("duration_s", lambda x: x.sum()/3600),
                                            mean_minutes=("duration_s", lambda x: x.mean()/60)).reindex(STYLES)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    labels = [STYLES[s][0] for s in summary.index]; colors = [STYLES[s][1] for s in summary.index]
    bars = axes[0].bar(labels, summary.arcs, color=colors)
    for bar, count in zip(bars, summary.arcs): axes[0].text(bar.get_x()+bar.get_width()/2, count, str(count), ha="center", va="bottom")
    axes[0].set_ylabel("Number of continuous arcs"); axes[0].set_title("Continuous-arc count")
    data = [arcs.loc[arcs.solution.eq(s), "duration_s"].div(60) for s in summary.index]
    box = axes[1].boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors): patch.set_facecolor(color)
    axes[1].set_ylabel("Arc duration (minutes)"); axes[1].set_title("Distribution of continuous-arc duration")
    fig.savefig(OUT / "continuous_arc_statistics.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    return summary


def plot_arc_timeline(arcs: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(15, 16), sharex=True, layout="constrained")
    for axis, solution in zip(axes, STYLES):
        subset = arcs[arcs.solution.eq(solution)].sort_values("satellite")
        satellites = sorted(subset.satellite.unique())
        ypos = {sat: i for i, sat in enumerate(satellites)}
        for row in subset.itertuples():
            axis.hlines(ypos[row.satellite], row.start, row.end, color="#0072B2" if row.system == "G" else "#E69F00", lw=2)
        axis.set_yticks(range(len(satellites))); axis.set_yticklabels(satellites, fontsize=7)
        axis.set_title(STYLES[solution][0]); axis.grid(axis="x", alpha=.25)
    axes[-1].set_xlabel("UTC time")
    fig.suptitle("Continuous observation arcs (blue: GPS; orange: BDS)")
    fig.savefig(OUT / "continuous_arc_timeline.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    primary, secondary = read_rinex(PRIMARY), read_rinex(SECONDARY)
    fused, _ = load_merge()(primary, secondary)
    solutions = {"primary": primary, "secondary": secondary, "fused": fused}
    epoch = count_records(solutions); epoch.to_csv(OUT / "satellite_count_by_epoch.csv", index=False)
    arcs = []
    for solution, data in solutions.items():
        value, cadence = continuous_arcs(data); value["solution"] = solution; arcs.append(value)
        print(f"{solution}: inferred sampling interval = {cadence:g} s")
    arcs = pd.concat(arcs, ignore_index=True); arcs.to_csv(OUT / "continuous_arc_by_satellite.csv", index=False)
    plot_counts(epoch); summary = plot_arc_statistics(arcs); plot_arc_timeline(arcs)
    summary.to_csv(OUT / "continuous_arc_summary.csv")
    print(summary.round(3)); print(f"Saved figures and tables to {OUT}")


if __name__ == "__main__": main()
