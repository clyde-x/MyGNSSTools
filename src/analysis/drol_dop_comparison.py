"""Compare observed-satellite DOP for Ant1, Ant2 and dual-antenna fusion."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Source layout is ``MyTools/src/analysis`` while shared helpers live in
# ``MyTools/lib``.  ``parents[1]`` points to ``src`` and therefore misses the
# actual library directory when this script is invoked directly.
LIB_DIR = Path(__file__).resolve().parents[2] / "lib"
sys.path.insert(0, str(LIB_DIR))
from DOPTool import (calculate_dop, calculate_rtn_matrix, interpolate_position,
                     iter_rinex_observed_prns, read_sp3_positions, rinex_day_tag)


def calculate_configuration(name, rinex_files, reference, gnss, receiver_id, systems):
    rows = []
    for raw_file in rinex_files:
        rinex = Path(raw_file); day = rinex_day_tag(rinex)
        if not rinex.is_file():
            raise FileNotFoundError(f"RINEX file not found: {rinex}")
        for time, observed in iter_rinex_observed_prns(rinex):
            timestamp = time.timestamp(); receiver = interpolate_position(reference, receiver_id, timestamp)
            usable_all = [prn for prn in observed if interpolate_position(gnss, prn, timestamp) is not None]
            for system in systems:
                prefixes = tuple(system["prefixes"])
                usable = [prn for prn in usable_all if prn.startswith(prefixes)]
                row = {"time": time, "day": day, "configuration": name, "system": system["label"], "rinex": rinex.name,
                       "observed_satellites": len(observed), "usable_satellites": len(usable),
                       "gps": sum(prn.startswith("G") for prn in usable), "bds": sum(prn.startswith("C") for prn in usable)}
                if receiver is None:
                    row["solution_status"] = "reference_orbit_unavailable"
                elif len(usable) < 4:
                    row["solution_status"] = "fewer_than_4_satellites"
                else:
                    result = calculate_dop(receiver, [interpolate_position(gnss, prn, timestamp) for prn in usable],
                                           calculate_rtn_matrix(receiver, reference, receiver_id, timestamp))
                    row["solution_status"] = "valid" if result is not None else "rank_deficient"
                    if result is not None:
                        row.update(result)
                rows.append(row)
    return rows


def statistics(valid: pd.DataFrame) -> pd.DataFrame:
    metrics = ["pdop", "gdop", "tdop", "rdop", "adop", "cdop", "usable_satellites"]
    result = valid.groupby(["configuration", "system", "day"])[metrics].agg(["mean", "median", lambda value: value.quantile(.95)])
    result.columns = [f"{metric}_{'p95' if stat == '<lambda_0>' else stat}" for metric, stat in result.columns]
    return result.reset_index()


def summary_text(frame: pd.DataFrame, metric: str, order: list[str]) -> str:
    values = frame.groupby("configuration")[metric].agg(["mean", lambda value: value.quantile(.95)])
    return "\n".join([f"{name}: {values.loc[name, 'mean']:.2f} / {values.loc[name, '<lambda_0>']:.2f}" for name in order])


def plot_results(valid: pd.DataFrame, output: Path, order: list[str], colors: dict[str, str], systems: list[dict], font_size: int, dpi: int):
    plt.rcParams.update({"font.family": "Arial", "font.size": font_size, "axes.labelsize": font_size + 1,
                         "axes.titlesize": font_size + 2, "legend.fontsize": font_size - 2, "savefig.dpi": dpi})
    for day, frame in valid.groupby("day"):
        fig, axes = plt.subplots(len(systems), 1, figsize=(13, 11), sharex=True, layout="constrained")
        for axis, system in zip(np.atleast_1d(axes), systems):
            subset = frame[frame.system.eq(system["label"])]
            ymax = subset.pdop.quantile(.99) if not subset.empty else 1
            for name in order:
                group = subset[subset.configuration.eq(name)]
                axis.plot(group.time, group.pdop, label=name, color=colors[name], lw=.75)
            axis.set_ylim(0, ymax); axis.set_ylabel("PDOP")
            axis.set_title(f"{system['label']} PDOP (99% display range)")
            axis.grid(True, alpha=.3); axis.legend(loc="upper left")
            axis.text(.985, .94, "Mean / P95\n" + summary_text(subset, "pdop", order), transform=axis.transAxes,
                      ha="right", va="top", fontsize=font_size - 2,
                      bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.5", alpha=.9))
        axes[-1].set_xlabel("GPS time")
        fig.suptitle(f"DOP comparison: antenna fusion and constellation combination ({day})")
        fig.savefig(output / f"dop_comparison_timeseries_{day}.png", bbox_inches="tight"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="DROL Ant1/Ant2/fused observed-satellite DOP comparison")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--font-size", type=int, default=15)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8")); section = config["dop"]
    output = Path(config["output_root"]); output.mkdir(parents=True, exist_ok=True)
    reference = read_sp3_positions(section["reference_orbits"])
    gnss = read_sp3_positions(section["precise_gnss_products"])
    receiver_id = section.get("receiver_satellite", "L02")
    configurations = section["configurations"]
    systems = section.get("systems", [{"label": "GPS+BDS", "prefixes": ["G", "C"]}])
    rows = []
    for item in configurations:
        rows.extend(calculate_configuration(item["label"], item["rinex_files"], reference, gnss, receiver_id, systems))
    epochs = pd.DataFrame(rows); epochs.to_csv(output / "dop_comparison_epoch_results.csv", index=False, float_format="%.6f")
    valid = epochs[epochs.solution_status.eq("valid")].copy()
    if valid.empty:
        raise RuntimeError("No valid DOP epochs. Check DOP lists and receiver_satellite.")
    table = statistics(valid); table.to_csv(output / "dop_comparison_summary.csv", index=False, float_format="%.4f")
    status = epochs.groupby(["configuration", "system", "day", "solution_status"]).size().rename("epochs").reset_index()
    status.to_csv(output / "dop_comparison_processing_status.csv", index=False)
    order = [item["label"] for item in configurations]
    default_colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    colors = {item["label"]: item.get("color") or default_colors[index % len(default_colors)] for index, item in enumerate(configurations)}
    plot_results(valid, output, order, colors, systems, args.font_size, args.dpi)
    print(table.groupby(["configuration", "system"])[["pdop_mean", "pdop_p95", "gdop_mean", "gdop_p95"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
