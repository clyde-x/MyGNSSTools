"""Count valid code observations by PRN and frequency in DROL GNSS-L1B RINEX."""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from plot.drol import plot_frequency_counts


def read_header(path: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="ascii", errors="ignore") as handle:
        for line in handle:
            if "END OF HEADER" in line:
                return result
            if "SYS / # / OBS TYPES" in line:
                system, total = line[0], int(line[3:6])
                result[system].extend(line[7:60].split())
                while len(result[system]) < total:
                    continuation = next(handle)
                    result[system].extend(continuation[7:60].split())
    raise ValueError(f"END OF HEADER not found: {path}")


def count_code_observations(path: Path) -> pd.DataFrame:
    types = read_header(path); counts: Counter = Counter(); in_header = True
    with path.open("r", encoding="ascii", errors="ignore") as handle:
        for line in handle:
            if in_header:
                in_header = "END OF HEADER" not in line
                continue
            if line.startswith(">") or len(line) < 3 or line[0] not in ("G", "C"):
                continue
            system, prn = line[0], line[:3].strip()
            for index, obs_type in enumerate(types[system]):
                if not obs_type.startswith("C"):
                    continue
                field = line[3 + 16 * index: 3 + 16 * (index + 1)]
                try:
                    value = float(field[:14])
                except ValueError:
                    continue
                if value != 0:
                    counts[(system, prn, obs_type)] += 1
    return pd.DataFrame([{"system": system, "prn": prn, "observation_type": obs_type, "valid_observations": value}
                         for (system, prn, obs_type), value in counts.items()])


def plot_system(frame: pd.DataFrame, system: str, output: Path, day: str, font_size: int, dpi: int):
    subset = frame[frame.system.eq(system)].copy()
    if subset.empty:
        return
    observation_types = sorted(subset.observation_type.unique())
    prns = sorted(subset.prn.unique(), key=lambda value: int(value[1:]))
    pivot = subset.pivot(index="prn", columns="observation_type", values="valid_observations").reindex(prns).fillna(0)
    plt.rcParams.update({"font.family": "Arial", "font.size": font_size, "axes.labelsize": font_size + 1,
                         "axes.titlesize": font_size + 2, "legend.fontsize": font_size - 2, "savefig.dpi": dpi})
    fig, ax = plt.subplots(figsize=(13, 6.4), layout="constrained")
    x = np.arange(len(prns)); width = min(.26, .78 / len(observation_types))
    palette = ["#E41A1C", "#0055CC", "#18A638", "#984EA3"]
    for index, obs_type in enumerate(observation_types):
        ax.bar(x + (index - (len(observation_types) - 1) / 2) * width, pivot[obs_type], width,
               label=obs_type, color=palette[index % len(palette)], edgecolor="none")
    system_name = {"G": "GPS", "C": "BDS"}[system]
    ax.set_xticks(x, [prn[1:] for prn in prns])
    ax.set_xlabel(f"{system_name} PRN"); ax.set_ylabel("Number of valid observations")
    ax.set_title(f"{system_name} L1B code-observation availability ({day})")
    ax.legend(title="Observation type", ncol=len(observation_types), loc="upper center")
    ax.grid(axis="y", alpha=.28); ax.set_axisbelow(True)
    fig.savefig(output / f"l1b_{system.lower()}_frequency_counts_{day}.png", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="DROL GNSS-L1B code-frequency availability by PRN")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--focus-day", required=True, help="YYYYMMDD")
    parser.add_argument("--font-size", type=int, default=15)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    doy = datetime.strptime(args.focus_day, "%Y%m%d").strftime("%Y%j")
    candidates = sorted(args.data_root.glob(f"*{doy}*.rnx"))
    if not candidates:
        raise FileNotFoundError(f"No GNSS-L1B RINEX found for {args.focus_day} in {args.data_root}")
    frames = []
    for path in candidates:
        table = count_code_observations(path)
        if not table.empty:
            table.insert(0, "file", path.name); frames.append(table)
    result = pd.concat(frames, ignore_index=True).groupby(["system", "prn", "observation_type"], as_index=False)["valid_observations"].sum()
    result.insert(0, "day", args.focus_day)
    result.to_csv(args.output / f"l1b_frequency_counts_{args.focus_day}.csv", index=False)
    plot_frequency_counts(result, "G", args.output, args.focus_day, args.font_size, args.dpi)
    plot_frequency_counts(result, "C", args.output, args.focus_day, args.font_size, args.dpi)
    print(result.groupby(["system", "observation_type"])["valid_observations"].sum().to_string())


if __name__ == "__main__":
    main()
