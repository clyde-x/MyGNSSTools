"""Visualize DROL ATT-L1B attitude changes and their GNSS-observation impact.

ATT-L1B quaternions are Qx, Qy, Qz, Qw and use GPS time.  The script plots
quaternion components and a frame-invariant angular-rate metric, then aligns
that metric with the retained RINEX satellite counts on the same GPS timeline.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_attitude(path: Path):
    rows = []
    for line in path.open(encoding="utf-8", errors="ignore"):
        values = line.split()
        if len(values) != 10:
            continue
        try:
            year, month, day, hour, minute = map(int, values[:5])
            second = float(values[5]); q = list(map(float, values[6:10]))
        except ValueError:
            continue
        rows.append((pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute, second=int(second))
                     + pd.to_timedelta(second % 1, unit="s"), *q))
    data = pd.DataFrame(rows, columns=["time", "qx", "qy", "qz", "qw"])
    q = data[["qx", "qy", "qz", "qw"]].to_numpy()
    q = q / np.linalg.norm(q, axis=1)[:, None]
    dots = np.abs(np.sum(q[1:] * q[:-1], axis=1)).clip(0, 1)
    elapsed = np.diff(data.time.astype("int64").to_numpy()) / 1e9
    data["angular_rate_deg_min"] = np.r_[np.nan, np.degrees(2 * np.arccos(dots)) / elapsed * 60]
    return data


def plot_day(attitude, observations, day: str, output: Path, font_size: int, dpi: int):
    att = attitude[attitude.time.dt.strftime("%Y%m%d").eq(day)].copy()
    if att.empty: return False
    plt.rcParams.update({"font.family": "Arial", "font.size": font_size, "axes.labelsize": font_size + 2,
                         "axes.titlesize": font_size + 3, "xtick.labelsize": font_size - 2, "ytick.labelsize": font_size - 1,
                         "legend.fontsize": font_size - 2, "savefig.dpi": dpi})
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.8), sharex=True, layout="constrained")
    for column, label in zip(["qx", "qy", "qz", "qw"], ["Qx", "Qy", "Qz", "Qw"]): axes[0].plot(att.time, att[column], lw=1.2, label=label)
    axes[0].set_ylabel("Quaternion component")
    axes[0].set_title(f"DROL attitude state on {day} (GPS time)")
    axes[0].legend(ncol=4)
    axes[1].plot(att.time, att.angular_rate_deg_min, color="#D55E00", lw=1.2)
    axes[1].set_ylabel("Angular rate (deg/min)")
    axes[1].set_xlabel("GPS time")
    fig.savefig(output / f"attitude_quaternion_rate_{day}.png", bbox_inches="tight")
    plt.close(fig)

    obs = observations[observations.day.astype(str).eq(day)].copy()
    if obs.empty: return True
    obs["minute"] = pd.to_datetime(obs.time).dt.floor("min")
    obs = obs.groupby(["solution", "minute"], as_index=False).total.mean()
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.8), sharex=True, layout="constrained")
    axes[0].plot(att.time, att.angular_rate_deg_min, color="#D55E00", lw=1.3)
    axes[0].set_ylabel("Angular rate (deg/min)")
    axes[0].set_title(f"Attitude motion and retained GNSS observations on {day}")
    styles = {"antenna_1": ("Antenna 1", "#0072B2"), "antenna_2": ("Antenna 2", "#D55E00"),
              "dual_antenna_fused": ("Dual-antenna fused", "#009E73")}
    for solution, (label, color) in styles.items():
        group = obs[obs.solution.eq(solution)]
        axes[1].plot(group.minute, group.total, lw=1.1, label=label, color=color)
    axes[1].set_ylabel("Retained satellites per epoch")
    axes[1].set_xlabel("GPS time")
    axes[1].legend(ncol=3, loc="upper right")
    fig.savefig(output / f"attitude_observation_relation_{day}.png", bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="DROL ATT-L1B and GNSS observation visualization")
    parser.add_argument("--attitude-root", type=Path, default=Path(r"D:\csu\dataset\DROL-GNSS\ATT-L1B"))
    parser.add_argument("--observation-csv", type=Path, default=Path(r"D:\csu\MyTools\analysis_output\DROL_batch\observation_epoch_availability.csv"))
    parser.add_argument("--output", type=Path, default=Path(r"D:\csu\MyTools\analysis_output\DROL_batch"))
    parser.add_argument("--days", nargs="*", default=["20241226", "20241227", "20241228", "20241229", "20241230"])
    parser.add_argument("--font-size", type=int, default=15)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    observations = pd.read_csv(args.observation_csv, parse_dates=["time"])
    records = []
    for day in args.days:
        doy = pd.to_datetime(day).strftime("%Y%j") + "0"
        path = args.attitude_root / f"DROL_{doy}.ATT"
        if not path.exists():
            records.append({"day": day, "status": "missing ATT"}); continue
        attitude = read_attitude(path)
        plot_day(attitude, observations, day, args.output, args.font_size, args.dpi)
        records.append({"day": day, "status": "ok", "epochs": len(attitude),
                        "max_rate_deg_min": attitude.angular_rate_deg_min.max(),
                        "p95_rate_deg_min": attitude.angular_rate_deg_min.quantile(.95)})
    pd.DataFrame(records).to_csv(args.output / "attitude_motion_summary.csv", index=False, float_format="%.6f")
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == "__main__": main()
