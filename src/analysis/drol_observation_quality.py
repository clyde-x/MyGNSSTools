"""Paper-ready availability analysis for corrected DROL RINEX observations.

Raw L1B is intentionally excluded because GPS L1 carrier phase can be tagged
as L2 in its header.  This script uses corrected Obs_dualfreq and the PCO-
corrected dual-antenna fusion products.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from plot.drol import plot_observation_availability


def parse_rinex(path: Path):
    """Return one record per epoch with GPS/BDS/total observed-satellite count."""
    records, current, gps, bds = [], None, 0, 0
    header = True
    for line in path.open("r", encoding="ascii", errors="ignore"):
        if header:
            if "END OF HEADER" in line: header = False
            continue
        if line.startswith(">"):
            if current is not None:
                records.append({"time": current, "gps": gps, "bds": bds, "total": gps + bds})
            parts = line[1:].split()
            second_value = float(parts[5])
            current = datetime(*map(int, parts[:5]), int(second_value),
                               round((second_value % 1) * 1_000_000))
            gps = bds = 0
        elif len(line) >= 3 and line[0] in ("G", "C") and line[1:3].isdigit():
            if line[0] == "G": gps += 1
            else: bds += 1
    if current is not None: records.append({"time": current, "gps": gps, "bds": bds, "total": gps + bds})
    return records


def day_tag(name):
    match = re.search(r"(2024\d{3}|2025\d{3})", name)
    if not match: return name
    return datetime.strptime(match.group(1), "%Y%j").strftime("%Y%m%d")


def plot_summary(summary, output: Path, font_size: int, dpi: int):
    plt.rcParams.update({"font.family": "Arial", "font.size": font_size, "axes.labelsize": font_size + 2,
                         "axes.titlesize": font_size + 3, "xtick.labelsize": font_size - 1, "ytick.labelsize": font_size - 1,
                         "legend.fontsize": font_size - 2, "savefig.dpi": dpi})
    order = ["antenna_1", "antenna_2", "dual_antenna_fused"]
    labels = ["Ant1", "Ant2", "Fused"]
    avg = summary.groupby("solution")[["mean_gps", "mean_bds", "mean_total", "below4_pct"]].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(10.5, 6.2), layout="constrained")
    bars = ax.bar(labels, avg["mean_gps"], label="GPS", color="#0072B2")
    ax.bar(labels, avg["mean_bds"], bottom=avg["mean_gps"], label="BDS", color="#E69F00")
    total_text = "\n".join(["Mean tracked satellites", *[f"{label}: {value:.2f}" for label, value in zip(labels, avg["mean_total"])]] )
    ax.text(0.02, 0.96, total_text, transform=ax.transAxes, va="top", fontsize=font_size - 1,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.5", alpha=0.88))
    ax.set_ylabel("Mean observed satellites per epoch")
    ax.set_title("Observation availability: single and dual antenna solutions")
    ax.legend()
    fig.savefig(output / "observation_availability_by_solution.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 6.2), layout="constrained")
    bars = ax.bar(labels, avg["below4_pct"], color=["#56B4E9", "#CC79A7", "#009E73"])
    for bar, value in zip(bars, avg["below4_pct"]): ax.text(bar.get_x()+bar.get_width()/2, value, f"{value:.2f}%", ha="center", va="bottom", fontsize=14)
    ax.set_ylabel("Epochs with fewer than 4 satellites (%)")
    ax.set_title("Low-visibility risk reduction by dual-antenna fusion")
    fig.savefig(output / "low_visibility_ratio_by_solution.png", bbox_inches="tight")
    plt.close(fig)


def plot_focus_day(epoch, output: Path, focus_day: str, styles=None, filename_prefix="observation"):
    focus = epoch[epoch.day.eq(focus_day)]
    if focus.empty: return
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, layout="constrained")
    styles = styles or {"antenna_1": ("Ant1", "#0072B2"), "antenna_2": ("Ant2", "#D55E00"),
                        "dual_antenna_fused": ("Fused", "#009E73")}
    for key, (label, color) in styles.items():
        group = focus[focus.solution.eq(key)]
        if group.empty:
            continue
        axes[0].plot(group.time, group.total, lw=.8, label=label, color=color)
        axes[1].plot(group.time, group.gps, lw=.8, label=label, color=color)
        axes[2].plot(group.time, group.bds, lw=.8, label=label, color=color)
    for axis, ylabel in zip(axes, ["Observed satellites", "GPS satellites", "BDS satellites"]):
        axis.set_ylabel(ylabel); axis.grid(True, alpha=.3)
    # plot a h-line at 4 satellites to highlight the low-visibility threshold
    for axis in axes: axis.axhline(4, color="0.5", lw=1.2, ls="--", alpha=.7)
    axes[0].set_title(f"DROL observation availability on {focus_day}")
    axes[0].legend(ncol=3, loc="upper right")
    for axis, column in zip(axes, ["total", "gps", "bds"]):
        means = focus.groupby("solution")[column].mean()
        mean_text = "\n".join(f"{styles[key][0]}: {means[key]:.2f}" for key in styles if key in means)
        axis.text(0.015, 0.96, mean_text, transform=axis.transAxes, va="top",
                  fontsize=plt.rcParams["font.size"] - 1,
                  bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.5", alpha=0.88))
    axes[-1].set_xlabel("UTC time")
    fig.savefig(output / f"{filename_prefix}_timeseries_{focus_day}.png", bbox_inches="tight")
    plt.close(fig)


def configured_dataediting_epochs(config_path: Path, names: list[str]):
    """Load DataEditing RINEX files selected by the fusion-batch configuration."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    roots = {key: Path(value) for key, value in config["paths"].items()}
    start = datetime.strptime(config["date_range"]["start"], "%Y%m%d")
    end = datetime.strptime(config["date_range"]["end"], "%Y%m%d")
    dates = [(start + pd.Timedelta(days=offset)).strftime("%Y%m%d")
             for offset in range((end - start).days + 1)]
    selected = [item for item in config["solutions"] if item["name"] in names]
    all_epochs = []
    for item in selected:
        for day in dates:
            config_name = f'{item["prefix"]}{day}'
            path = roots["tempdata_root"] / item["batch"] / config_name / f"DRO_DataEditing_Obs_{day}.rnx"
            if not path.exists():
                continue
            for record in parse_rinex(path):
                all_epochs.append({"solution": item["name"], "day": day, "file": str(path), **record})
    styles = {item["name"]: (item.get("label", item["name"]), item.get("color", "#0072B2"))
              for item in selected}
    return pd.DataFrame(all_epochs).sort_values(["solution", "time"]), styles


def main():
    parser = argparse.ArgumentParser(description="DROL corrected-RINEX observation availability")
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\csu\dataset\DROL-GNSS"))
    parser.add_argument("--output", type=Path, default=Path(r"D:\csu\MyTools\analysis_output\DROL_batch"))
    parser.add_argument("--focus-day", default="20241228")
    parser.add_argument("--font-size", type=int, default=15)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--fusion-batch-config", type=Path,
                        help="Use DataEditing RINEX files listed by this fusion-batch configuration.")
    parser.add_argument("--solutions", nargs="+", default=["single_ant1", "single_ant2", "priority"],
                        help="Solution names used with --fusion-batch-config.")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if args.fusion_batch_config:
        epoch, styles = configured_dataediting_epochs(args.fusion_batch_config, args.solutions)
        prefix = "dataediting_observation"
    else:
        collections = {
            "antenna_1": sorted((args.data_root / "Obs_dualfreq").glob("DL0100*.rnx")),
            "antenna_2": sorted((args.data_root / "Obs_dualfreq").glob("DL0200*.rnx")),
            "dual_antenna_fused": sorted((args.data_root / "Obs_ant3").glob("DROL_*_AfterPCO_Merge2Ant.rnx")),
        }
        all_epochs = []
        for solution, files in collections.items():
            for path in files:
                for record in parse_rinex(path): all_epochs.append({"solution": solution, "day": day_tag(path.name), "file": path.name, **record})
        epoch = pd.DataFrame(all_epochs).sort_values(["solution", "time"])
        styles = None
        prefix = "observation"
    if epoch.empty:
        raise RuntimeError("No observation epochs found for the selected input.")
    summary = epoch.groupby(["solution", "day", "file"], as_index=False).agg(
        epochs=("total", "size"), mean_gps=("gps", "mean"), mean_bds=("bds", "mean"), mean_total=("total", "mean"),
        min_total=("total", "min"), below4_pct=("total", lambda x: 100 * (x < 4).mean()))
    epoch.to_csv(args.output / f"{prefix}_epoch_availability.csv", index=False)
    summary.to_csv(args.output / f"{prefix}_availability.csv", index=False, float_format="%.6f")
    # plot_summary(summary, args.output, args.font_size, args.dpi); 
    plot_observation_availability(epoch, args.output, args.focus_day, styles, prefix)
    print(summary.groupby("solution")[["mean_total", "below4_pct"]].mean().round(3))
    print(f"Saved observation tables and figures to {args.output}")


if __name__ == "__main__": main()
