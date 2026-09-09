"""Parse and plot inferred RTN force components from headerless FORCE_RTN.DAT."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from plot.drol import plot_force_rtn

LIB_DIR = ROOT_DIR / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))
from TimeSystem import TimeSystem


COLUMN_GROUPS = {
    "Central gravity": ("central_r", "central_t", "central_n"),
    "Non-spherical gravity": ("nonspherical_r", "nonspherical_t", "nonspherical_n"),
    "Atmospheric drag": ("drag_r", "drag_t", "drag_n"),
    "Solar radiation pressure": ("srp_r", "srp_t", "srp_n"),
}
COMPONENTS = ("R", "T", "N")


def parse_force_rtn(path: Path) -> pd.DataFrame:
    """Parse lines containing 13 values: MJD then 4 inferred RTN triplets."""
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            payload = line.split("FORCE_RTN:", 1)[-1] if "FORCE_RTN:" in line else line
            fields = payload.split()
            if len(fields) < 13:
                continue
            try:
                values = [float(value) for value in fields[:13]]
            except ValueError:
                continue
            rows.append(values)
    if not rows:
        raise ValueError(f"No FORCE_RTN records found in {path}")
    columns = ["mjd", *[column for group in COLUMN_GROUPS.values() for column in group]]
    frame = pd.DataFrame(rows, columns=columns).drop_duplicates("mjd").sort_values("mjd")
    # Use the project's canonical time converter; TimeSystem currently assumes
    # a continuous UTC-like scale and does not apply leap-second corrections.
    frame["utc"] = [TimeSystem().from_mjd(float(mjd)).datetime for mjd in frame.mjd]
    return frame


def decimate(frame: pd.DataFrame, max_points: int) -> pd.DataFrame:
    step = max(1, int(np.ceil(len(frame) / max_points)))
    return frame.iloc[::step]


def plot_components(frame: pd.DataFrame, output: Path, font_size: int, dpi: int, max_points: int) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": font_size, "axes.labelsize": font_size + 1,
                         "axes.titlesize": font_size + 1, "legend.fontsize": font_size - 2, "savefig.dpi": dpi})
    shown = decimate(frame, max_points)
    colors = ["#0072B2", "#D55E00", "#009E73"]
    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True, layout="constrained")
    for axis, (name, columns) in zip(axes, COLUMN_GROUPS.items()):
        for component, column, color in zip(COMPONENTS, columns, colors):
            axis.plot(shown.utc, shown[column], lw=.65, label=component, color=color)
        mean_norm = np.linalg.norm(frame[list(columns)].to_numpy(), axis=1).mean()
        axis.text(.985, .92, f"Mean |a|: {mean_norm:.3e} m/s²", transform=axis.transAxes, ha="right", va="top",
                  fontsize=font_size - 2, bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.5", alpha=.88))
        axis.set_ylabel("Acceleration (m/s²)"); axis.set_title(name); axis.grid(True, alpha=.28); axis.legend(ncol=3, loc="upper left")
    axes[-1].set_xlabel("UTC time (from MJD)")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    fig.suptitle("FORCE_RTN.DAT: inferred RTN force components (header absent)")
    fig.savefig(output / "force_rtn_inferred_components.png", bbox_inches="tight"); plt.close(fig)


def plot_magnitudes(frame: pd.DataFrame, output: Path, font_size: int, dpi: int, max_points: int) -> None:
    shown = decimate(frame, max_points)
    fig, ax = plt.subplots(figsize=(14, 6.2), layout="constrained")
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for (name, columns), color in zip(COLUMN_GROUPS.items(), colors):
        magnitude = np.linalg.norm(shown[list(columns)].to_numpy(), axis=1)
        ax.semilogy(shown.utc, magnitude, lw=.75, label=name, color=color)
    ax.set_title("Inferred force magnitudes in RTN frame (header absent)")
    ax.set_ylabel("Acceleration magnitude (m/s²)"); ax.set_xlabel("UTC time (from MJD)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M")); ax.grid(True, which="both", alpha=.28); ax.legend(ncol=2)
    fig.savefig(output / "force_rtn_inferred_magnitudes.png", bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Headerless FORCE_RTN.DAT inferred RTN-force visualization")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font-size", type=int, default=15)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--max-points", type=int, default=12000)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    frame = parse_force_rtn(args.input)
    frame.to_csv(args.output / "force_rtn_inferred_epoch_values.csv", index=False, float_format="%.12e")
    summary_rows = []
    for name, columns in COLUMN_GROUPS.items():
        values = frame[list(columns)].to_numpy()
        summary_rows.append({"inferred_force_group": name, "mean_norm_mps2": np.linalg.norm(values, axis=1).mean(),
                             "p95_norm_mps2": np.quantile(np.linalg.norm(values, axis=1), .95),
                             "max_norm_mps2": np.linalg.norm(values, axis=1).max(),
                             "r_mean_mps2": values[:, 0].mean(), "t_mean_mps2": values[:, 1].mean(), "n_mean_mps2": values[:, 2].mean()})
    pd.DataFrame(summary_rows).to_csv(args.output / "force_rtn_inferred_summary.csv", index=False, float_format="%.12e")
    plot_force_rtn(frame, COLUMN_GROUPS, args.output, args.font_size, args.dpi, args.max_points)
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
