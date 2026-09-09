"""Trial implementation of 50-s valid-arc, satellite-wise primary-priority fusion.

It deliberately keeps the production algorithm unchanged.  Results are written
under MyTools/test and the arc analysis reuses drol_arc_analysis.py functions.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

TOOLS = Path(r"D:\csu\MyTools")
LIB = TOOLS / "lib"; SRC = TOOLS / "src"
sys.path[:0] = [str(LIB), str(SRC)]
import Merge_TwoAnts_Rinex as merger
import drol_arc_analysis as arc_analysis

PRIMARY = Path(r"D:\csu\dataset\DROL-GNSS\Obs_pco\DL0200LEO_20243600_PCO.rnx")
SECONDARY = Path(r"D:\csu\dataset\DROL-GNSS\Obs_pco\DL0100LEO_20243600_PCO.rnx")
OUT = TOOLS / "test" / "valid_arc_50s_analysis"
MERGED_FILE = OUT / "DL0300LEO_20243600_valid_arc50_primary_priority.rnx"
MIN_ARC_SECONDS = 50.0
GAP_FACTOR = 1.5


def sampling_interval(data):
    epochs = sorted(data)
    gaps = [(b - a).total_seconds() for a, b in zip(epochs, epochs[1:]) if b > a]
    return min(gaps)


def valid_arc_membership(data, min_duration_s):
    """Return every (satellite, epoch) belonging to a sufficiently long arc."""
    by_sat = defaultdict(list)
    for epoch, info in data.items():
        for satellite in info["observations"]:
            by_sat[satellite].append(epoch)
    dt = sampling_interval(data)
    members = defaultdict(set)
    rows = []
    for satellite, times in by_sat.items():
        times = sorted(times); start_i = 0
        for i in range(1, len(times) + 1):
            is_break = i == len(times) or (times[i] - times[i - 1]).total_seconds() > GAP_FACTOR * dt
            if not is_break:
                continue
            start, end = times[start_i], times[i - 1]
            # Inclusive coverage: a single 10-s sample represents 10 seconds.
            duration = (end - start).total_seconds() + dt
            if duration >= min_duration_s:
                members[satellite].update(times[start_i:i])
                rows.append({"satellite": satellite, "start": start, "end": end,
                             "epochs": i - start_i, "duration_s": duration})
            start_i = i
    return members, pd.DataFrame(rows), dt


def fuse(primary, secondary, primary_ok, secondary_ok):
    result, sources = {}, {}
    for epoch in sorted(set(primary) | set(secondary)):
        observations = {}
        for satellite in set(primary.get(epoch, {}).get("observations", {})) | set(secondary.get(epoch, {}).get("observations", {})):
            if epoch in primary_ok.get(satellite, set()):
                observations[satellite] = primary[epoch]["observations"][satellite]
                sources[(epoch, satellite)] = "primary"
            elif epoch in secondary_ok.get(satellite, set()):
                observations[satellite] = secondary[epoch]["observations"][satellite]
                sources[(epoch, satellite)] = "secondary"
        if observations:
            result[epoch] = {"time": epoch, "flag": "0", "num_satellites": len(observations), "observations": observations}
    return result, sources


def filtered(data, membership):
    return {epoch: {**info, "observations": {sat: obs for sat, obs in info["observations"].items() if epoch in membership.get(sat, set())}}
            for epoch, info in data.items()}


def epoch_summary(primary, secondary, fused):
    rows = []
    for epoch in sorted(set(primary) | set(secondary)):
        a = set(primary.get(epoch, {}).get("observations", {})); b = set(secondary.get(epoch, {}).get("observations", {}))
        c = set(fused.get(epoch, {}).get("observations", {}))
        rows.append({"time": epoch, "primary_valid": len(a), "secondary_valid": len(b),
                     "theoretical_union": len(a | b), "fused": len(c),
                     "equals_union": c == a | b})
    return pd.DataFrame(rows)


def plot_satellite_counts(primary, secondary, fused):
    """Plot per-epoch satellite counts after the 50-s valid-arc screening."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True, layout="constrained")
    sources = {"Primary (DL0200)": (primary, "#0072B2"),
               "Secondary (DL0100)": (secondary, "#D55E00"),
               "Fused (valid-arc primary priority)": (fused, "#009E73")}
    columns = [("total", "Observed satellites"), ("gps", "GPS satellites"), ("bds", "BDS satellites")]
    for label, (data, color) in sources.items():
        times = sorted(data)
        counts = {"total": [], "gps": [], "bds": []}
        for epoch in times:
            satellites = data[epoch]["observations"]
            counts["total"].append(len(satellites))
            counts["gps"].append(sum(s.startswith("G") for s in satellites))
            counts["bds"].append(sum(s.startswith("C") for s in satellites))
        for axis, (column, _) in zip(axes, columns):
            axis.plot(times, counts[column], lw=.7, color=color, label=label)
    for axis, (_, ylabel) in zip(axes, columns):
        axis.set_ylabel(ylabel); axis.grid(True, alpha=.28)
    axes[0].set_title("Satellite availability after 50-s valid-arc screening")
    axes[0].legend(ncol=3, loc="upper right")
    axes[-1].set_xlabel("UTC time")
    fig.savefig(OUT / "valid_arc50_satellite_count_timeseries.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def analyse_arcs():
    inputs = {"antenna_1": PRIMARY, "antenna_2": SECONDARY, "dual_antenna_fused": MERGED_FILE}
    rows = []
    for solution, path in inputs.items():
        rows.extend({"solution": solution, **row} for row in arc_analysis.read_tracking(path))
    arcs = arc_analysis.build_arcs(pd.DataFrame(rows), GAP_FACTOR)
    arcs.to_csv(OUT / "continuous_tracking_arcs.csv", index=False)
    stats = arcs.groupby(["solution", "system"], as_index=False).agg(
        arcs=("duration_s", "size"), mean_min=("duration_s", lambda x: x.mean()/60),
        median_min=("duration_s", lambda x: x.median()/60), p95_min=("duration_s", lambda x: x.quantile(.95)/60),
        over10min_pct=("duration_s", lambda x: 100 * (x >= 600).mean()))
    stats.to_csv(OUT / "continuous_arc_statistics.csv", index=False, float_format="%.6f")
    arc_analysis.plot_distribution(arcs, OUT, font_size=15, dpi=300)
    arc_analysis.plot_timeline(arcs, "2024-12-25", OUT, font_size=15, dpi=300)
    return stats


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    header, primary = merger.read_rinex_obs(PRIMARY, system="MIX")
    _, secondary = merger.read_rinex_obs(SECONDARY, system="MIX")
    primary_ok, primary_arcs, primary_dt = valid_arc_membership(primary, MIN_ARC_SECONDS)
    secondary_ok, secondary_arcs, secondary_dt = valid_arc_membership(secondary, MIN_ARC_SECONDS)
    fused, sources = fuse(primary, secondary, primary_ok, secondary_ok)
    merger.write_merged_rinex(MERGED_FILE, header, fused)
    valid_primary, valid_secondary = filtered(primary, primary_ok), filtered(secondary, secondary_ok)
    epochs = epoch_summary(valid_primary, valid_secondary, fused)
    epochs.to_csv(OUT / "valid_arc_fusion_epoch_summary.csv", index=False)
    plot_satellite_counts(valid_primary, valid_secondary, fused)
    pd.concat([primary_arcs.assign(source="primary"), secondary_arcs.assign(source="secondary")]).to_csv(OUT / "preprocessed_valid_arcs.csv", index=False)
    arc_stats = analyse_arcs()
    print(f"Sampling interval: primary={primary_dt:g} s, secondary={secondary_dt:g} s; min arc={MIN_ARC_SECONDS:g} s")
    print(f"Selected observations: {dict(pd.Series(list(sources.values())).value_counts())}")
    print("Filtered-epoch comparison:")
    print(epochs[["primary_valid", "secondary_valid", "theoretical_union", "fused"]].mean().round(3))
    print(f"Fused equals filtered union: {epochs.equals_union.sum()} / {len(epochs)} epochs")
    print("Carrier-tracking arc statistics (from drol_arc_analysis.py):")
    print(arc_stats.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"Saved trial fusion and arc analysis to {OUT}")


if __name__ == "__main__":
    main()
