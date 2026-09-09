"""Config-driven batch validation for DROL fusion POD solutions.

The tool evaluates every configured solution/day from CSUPODS ``outputdata``,
``LOG`` and ``TempData``.  It reports PRE orbit agreement where an external
PRE exists, plus RDOD posterior phase residuals and DataEditing availability.
Results are written as JSON, CSV tables, and PNG figures.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

MYTOOLS_ROOT = Path(r"D:\csu\MyTools")
for module_dir in (MYTOOLS_ROOT, MYTOOLS_ROOT / "lib", MYTOOLS_ROOT / "src" / "analysis"):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
import analysis
from plot.residual import plot_posterior_residuals


def parse_sp3(path: Path) -> dict[datetime, np.ndarray]:
    """Read the one-spacecraft positions from an SP3-compatible product."""
    positions: dict[datetime, np.ndarray] = {}
    epoch: datetime | None = None
    for line in path.open(encoding="utf-8", errors="ignore"):
        if line.startswith("*"):
            fields = line[1:].split()
            if len(fields) < 6:
                continue
            second = float(fields[5])
            epoch = datetime(*map(int, fields[:5])) + timedelta(seconds=second)
        elif epoch and line.startswith("P"):
            fields = line.split()
            try:
                xyz_km = [float(value) for value in fields[1:4]]
            except (ValueError, IndexError):
                continue
            if max(map(abs, xyz_km)) < 999999.0:
                positions[epoch] = np.asarray(xyz_km)
    return positions


def pre_path(date_text: str, pre_root: Path) -> Path | None:
    day = datetime.strptime(date_text, "%Y%m%d")
    tag = f"{day.year}{day.timetuple().tm_yday:03d}0"
    path = pre_root / f"REDA_{tag}_DL00.PRE"
    return path if path.exists() else None


def orbit_comparison(solution: Path, reference: Path) -> tuple[dict[str, float | int | None], list[tuple[datetime, float]]]:
    """Return daily orbit statistics and the 3D error at every common epoch."""
    tested, pre = parse_sp3(solution), parse_sp3(reference)
    common = sorted(set(tested) & set(pre))
    if len(common) < 3:
        return ({"pre_common_epochs": len(common), "pre_3d_rms_m": None,
                 "pre_3d_p95_m": None, "pre_r_rms_m": None,
                 "pre_t_rms_m": None, "pre_n_rms_m": None}, [])
    delta_m = np.asarray([tested[time] - pre[time] for time in common]) * 1000.0
    reference_xyz = np.asarray([pre[time] for time in common])
    velocity = np.gradient(reference_xyz, axis=0)
    r_hat = reference_xyz / np.linalg.norm(reference_xyz, axis=1)[:, None]
    n_hat = np.cross(reference_xyz, velocity)
    n_hat /= np.linalg.norm(n_hat, axis=1)[:, None]
    t_hat = np.cross(n_hat, r_hat)
    rtn = np.column_stack((np.sum(delta_m * r_hat, axis=1),
                           np.sum(delta_m * t_hat, axis=1),
                           np.sum(delta_m * n_hat, axis=1)))
    norm = np.linalg.norm(delta_m, axis=1)
    metrics = {
        "pre_common_epochs": len(common),
        "pre_3d_rms_m": float(np.sqrt(np.mean(norm ** 2))),
        "pre_3d_p95_m": float(np.quantile(norm, 0.95)),
        "pre_r_rms_m": float(np.sqrt(np.mean(rtn[:, 0] ** 2))),
        "pre_t_rms_m": float(np.sqrt(np.mean(rtn[:, 1] ** 2))),
        "pre_n_rms_m": float(np.sqrt(np.mean(rtn[:, 2] ** 2))),
    }
    return metrics, list(zip(common, norm.tolist()))


def residual_metrics(path: Path) -> dict[str, float | int | None]:
    phase: list[float] = []
    counts: list[int] = []
    for line in path.open(encoding="utf-8", errors="ignore"):
        fields = line.split()
        if len(fields) < 8:
            continue
        try:
            number = int(fields[7])
        except ValueError:
            continue
        counts.append(number)
        for index in range(number):
            start = 8 + 5 * index
            if start + 4 >= len(fields):
                break
            try:
                phase.append(float(fields[start + 3]))
            except ValueError:
                pass
    values = np.asarray(phase)
    if not len(values):
        return {"rdod_epochs": len(counts), "rdod_observations": 0,
                "rdod_mean_sats": None, "phase_rms_m": None,
                "phase_p95_abs_m": None}
    return {"rdod_epochs": len(counts), "rdod_observations": len(values),
            "rdod_mean_sats": float(np.mean(counts)),
            "phase_rms_m": float(np.sqrt(np.mean(values ** 2))),
            "phase_p95_abs_m": float(np.quantile(np.abs(values), 0.95))}


def rinex_epoch_counts(path: Path) -> list[tuple[datetime, int]]:
    """Read RINEX-3 epoch timestamps and satellite counts without parsing observations."""
    counts: list[tuple[datetime, int]] = []
    for line in path.open(encoding="utf-8", errors="ignore"):
        if not line.startswith(">"): 
            continue
        fields = line.split()
        try:
            # RINEX-3 epoch tokens: > yyyy mm dd hh mm ss flag nsat ...
            epoch = datetime(*map(int, fields[1:6])) + timedelta(seconds=float(fields[6]))
            counts.append((epoch, int(fields[8])))
        except (IndexError, TypeError, ValueError):
            continue
    return counts


def editing_metrics(path: Path) -> dict[str, float | int | None]:
    counts = [count for _, count in rinex_epoch_counts(path)]
    return {"editing_epochs": len(counts), "editing_observations": sum(counts),
            "editing_mean_sats": float(np.mean(counts)) if counts else None,
            "editing_epochs_ge5": sum(value >= 5 for value in counts)}


def dates(config: dict) -> list[str]:
    start = datetime.strptime(config["date_range"]["start"], "%Y%m%d")
    end = datetime.strptime(config["date_range"]["end"], "%Y%m%d")
    return [(start + timedelta(days=offset)).strftime("%Y%m%d")
            for offset in range((end - start).days + 1)]


def safe_median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def summarize(rows: list[dict], min_epochs: int) -> list[dict]:
    result = []
    for name, group in defaultdict(list, {name: [row for row in rows if row["solution"] == name]
                                          for name in {row["solution"] for row in rows}}).items():
        complete = [row for row in group if (row.get("editing_epochs") or 0) >= min_epochs]
        pre_rows = [row for row in group if row.get("pre_3d_rms_m") is not None]
        result.append({
            "solution": name,
            "completed_days": sum(row["status"] == "ok" for row in group),
            "pre_days": len(pre_rows),
            "pre_3d_rms_median_m": safe_median([row["pre_3d_rms_m"] for row in pre_rows]),
            "pre_3d_rms_worst_m": max((row["pre_3d_rms_m"] for row in pre_rows), default=None),
            "phase_rms_median_m": safe_median([row["phase_rms_m"] for row in complete if row.get("phase_rms_m") is not None]),
            "editing_observations_median": safe_median([row["editing_observations"] for row in complete]),
            "editing_epochs_median": safe_median([row["editing_epochs"] for row in complete]),
        })
    return sorted(result, key=lambda row: row["solution"])


def plot_multiday_residuals(residuals: dict[str, dict], solutions: list[dict],
                            output: Path, dpi: int, options: dict | None = None) -> None:
    """Write a time-scatter and a sky-view figure for every solution."""
    target = output / "residual_multiday"
    target.mkdir(parents=True, exist_ok=True)
    options = options or {}
    for solution in solutions:
        name = solution["name"]
        data = residuals.get(name, {})
        if not data:
            continue
        result = plot_posterior_residuals(
            data, f"RDOD_{name}_all_dates", target, dpi=dpi,
            scatter_size=float(options.get("scatter_size", 2.0)),
            alpha=float(options.get("alpha", .42)),
            time_clip_percent=float(options.get("time_clip_percent", 0.0)),
            sky_color_percentile=float(options.get("sky_color_percentile", 95.0)),
            cmap=str(options.get("cmap", "coolwarm")),
        )
        print(f"Saved posterior residual figures for {name}: {result.time_series_path.name}, {result.sky_plot_path.name}")


def write_orbit_epoch_csv(orbit_epochs: dict[str, list[tuple[datetime, float]]],
                          solutions: list[dict], output: Path, clip_ymax_cm: float) -> None:
    """Persist every matched PRE epoch; plot clipping never discards source data."""
    labels = {item["name"]: item.get("label", item["name"]) for item in solutions}
    target = output / "pre_3d_epoch_rms_all_dates.csv"
    with target.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=("solution", "label", "epoch_utc",
                                                     "pre_3d_rms_m", "pre_3d_rms_cm",
                                                     "shown_in_clipped_plot"))
        writer.writeheader()
        for name in labels:
            for epoch, error_m in sorted(orbit_epochs.get(name, []), key=lambda item: item[0]):
                error_cm = error_m * 100.0
                writer.writerow({"solution": name, "label": labels[name],
                                 "epoch_utc": epoch.isoformat(sep=" "),
                                 "pre_3d_rms_m": f"{error_m:.9f}",
                                 "pre_3d_rms_cm": f"{error_cm:.6f}",
                                 "shown_in_clipped_plot": error_cm <= clip_ymax_cm})


def write_residual_csv(residuals: dict[str, dict], solutions: list[dict], output: Path,
                       phase_clip_mm: float) -> None:
    """Persist the observation-level RDOD residuals used by the four-panel plots."""
    labels = {item["name"]: item.get("label", item["name"]) for item in solutions}
    target = output / "rdod_residuals_all_dates.csv"
    epochs = sorted({epoch.datetime for data in residuals.values() for epoch in data})
    epoch_index = {epoch: index + 1 for index, epoch in enumerate(epochs)}
    fields = ("solution", "label", "epoch_utc", "epoch_index", "satellite", "elevation_deg", "azimuth_deg",
              "prefitL_m", "prefitC_m", "shown_in_phase_scatter")
    with target.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name in labels:
            for epoch, satellites in sorted(residuals.get(name, {}).items(), key=lambda item: item[0]):
                for satellite, values in sorted(satellites.items()):
                    writer.writerow({
                        "solution": name,
                        "label": labels[name],
                        "epoch_utc": epoch.datetime.isoformat(sep=" "),
                        "epoch_index": epoch_index[epoch.datetime], "satellite": satellite,
                        "elevation_deg": values.get("ele"),
                        "azimuth_deg": values.get("azi"),
                        "prefitL_m": values.get("prefitL"),
                        "prefitC_m": values.get("prefitC"),
                        "shown_in_phase_scatter": (values.get("prefitL") is not None and
                                                    abs(values["prefitL"]) * 1000 <= phase_clip_mm),
                    })


def plot_posterior_phase_residual_scatter(residuals: dict[str, dict], solutions: list[dict],
                                          output: Path, dpi: int, clip_ymax_mm: float) -> None:
    """Plot each observation residual against a common sequential epoch index."""
    all_epochs = sorted({epoch.datetime for data in residuals.values() for epoch in data})
    if not all_epochs:
        return
    epoch_index = {epoch: index + 1 for index, epoch in enumerate(all_epochs)}
    fig, axes = plt.subplots(len(solutions), 1, figsize=(11.5, 10.5), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, solution in zip(axes, solutions):
        name = solution["name"]
        x_values, residual_mm = [], []
        for epoch, satellites in sorted(residuals.get(name, {}).items(), key=lambda item: item[0]):
            for values in satellites.values():
                phase = values.get("prefitL")
                if phase is not None:
                    x_values.append(epoch_index[epoch.datetime])
                    residual_mm.append(float(phase) * 1000.0)
        if residual_mm:
            y = np.asarray(residual_mm)
            x = np.asarray(x_values)
            visible = np.abs(y) <= clip_ymax_mm
            axis.scatter(x[visible], y[visible], s=1.5, alpha=.28, linewidths=0,
                         color=solution.get("color"), rasterized=True)
            axis.text(.01, .91, f"{solution.get('label', name)}: {visible.sum()}/{len(y)} shown",
                      transform=axis.transAxes, fontsize=8,
                      bbox=dict(boxstyle="round", facecolor="white", alpha=.8))
        axis.axhline(0, color="0.35", linewidth=.6)
        axis.set_ylabel("Phase residual\n(mm)")
        axis.set_ylim(-clip_ymax_mm, clip_ymax_mm)
        axis.grid(alpha=.25)
    axes[0].set_title("DROL fusion POD: posterior carrier-phase residuals")
    axes[-1].set_xlabel("Sequential epoch index (all residual epochs)")
    fig.text(.995, .995, f"Points with |residual| > {clip_ymax_mm:g} mm omitted; full data retained in CSV",
             ha="right", va="top", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(output / "posterior_phase_residual_epoch_scatter_all_dates.png", dpi=dpi)
    plt.close(fig)


def write_satellite_count_csv(rows: list[dict], solutions: list[dict], output: Path) -> None:
    """Write long-form data and one epoch-aligned wide CSV for each observation stage."""
    fields = ("solution", "label", "date", "stage", "epoch_utc", "satellite_count")
    target = output / "observation_satellite_counts_all_dates.csv"
    with target.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key != "epoch"})

    names = [solution["name"] for solution in solutions]
    for stage, filename in (("raw_fusion", "raw_fusion_satellite_counts_by_epoch.csv"),
                            ("dataediting", "dataediting_satellite_counts_by_epoch.csv")):
        values: dict[datetime, dict[str, int]] = defaultdict(dict)
        for row in rows:
            if row["stage"] == stage:
                values[row["epoch"]][row["solution"]] = row["satellite_count"]
        with (output / filename).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=("epoch_utc", *names))
            writer.writeheader()
            for epoch in sorted(values):
                writer.writerow({"epoch_utc": epoch.isoformat(sep=" "), **values[epoch]})


def plot_satellite_counts(rows: list[dict], solutions: list[dict], output: Path, dpi: int) -> None:
    """Plot all raw and DataEditing epoch counts on one multi-day time axis."""
    if not rows:
        return
    labels = {item["name"]: item.get("label", item["name"]) for item in solutions}
    colors = {item["name"]: item.get("color") for item in solutions}
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
    for axis, stage, title in zip(axes, ("raw_fusion", "dataediting"),
                                  ("Raw fused observations", "After DataEditing")):
        for name in labels:
            values = [row for row in rows if row["stage"] == stage and row["solution"] == name]
            if values:
                axis.scatter([row["epoch"] for row in values], [row["satellite_count"] for row in values],
                             s=4, alpha=.42, linewidths=0, color=colors[name], label=labels[name])
        axis.set_ylabel("Satellite count")
        axis.set_title(title)
        axis.grid(alpha=.3)
    axes[0].legend(ncol=3, loc="upper right")
    axes[-1].set_xlabel("Date (all available epochs)")
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%Y"))
    fig.suptitle("DROL fusion: satellite availability before and after DataEditing")
    fig.tight_layout()
    fig.savefig(output / "observation_satellite_counts_all_dates.png", dpi=dpi)
    plt.close(fig)


def plot_results(rows: list[dict], orbit_epochs: dict[str, list[tuple[datetime, float]]],
                 solutions: list[dict], output: Path, dpi: int, clip_ymax_cm: float) -> None:
    labels = {item["name"]: item.get("label", item["name"]) for item in solutions}
    colors = {item["name"]: item.get("color") for item in solutions}
    if any(orbit_epochs.values()):
        # A common, time-ordered epoch index keeps all solution points on one
        # horizontal time base while avoiding day-level date tick aggregation.
        common_epochs = sorted({epoch for points in orbit_epochs.values() for epoch, _ in points})
        epoch_to_index = {epoch: index + 1 for index, epoch in enumerate(common_epochs)}
        fig, ax = plt.subplots(figsize=(11.5, 5.8))
        for name in labels:
            values = sorted(orbit_epochs.get(name, []), key=lambda item: item[0])
            if values:
                times, errors_m = zip(*values)
                errors_cm = np.asarray(errors_m) * 100.0
                visible = errors_cm <= clip_ymax_cm
                epoch_index = np.asarray([epoch_to_index[epoch] for epoch in times])
                ax.scatter(epoch_index[visible], errors_cm[visible], s=5, alpha=.48,
                           linewidths=0,
                           label=f"{labels[name]} ({visible.sum()}/{len(visible)})",
                           color=colors[name])
        ax.set(xlabel="Sequential epoch index (all reference-covered epochs)", ylabel="PRE 3D RMS / error (cm)",
               title="DROL fusion POD: per-epoch external PRE 3D RMS")
        ax.set_ylim(0, clip_ymax_cm)
        ax.text(.99, .98, f"Points > {clip_ymax_cm:g} cm omitted from plot\nLegend: shown / total",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=.8))
        ax.grid(alpha=.3); ax.legend(ncol=2); fig.tight_layout()
        fig.savefig(output / "pre_3d_epoch_rms_scatter_all_dates.png", dpi=dpi); plt.close(fig)

    phase_rows = [row for row in rows if row.get("phase_rms_m") is not None]
    if phase_rows:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for name in labels:
            values = sorted((row for row in phase_rows if row["solution"] == name), key=lambda row: row["date"])
            if values:
                ax.plot([row["date"][4:] for row in values], [1000 * row["phase_rms_m"] for row in values],
                        marker="o", label=labels[name], color=colors[name])
        ax.set(xlabel="Date (MMDD)", ylabel="RDOD phase RMS (mm)", title="DROL fusion POD: posterior phase residual")
        ax.grid(alpha=.3); ax.legend(ncol=2); fig.tight_layout()
        fig.savefig(output / "rdod_phase_rms_by_date.png", dpi=dpi); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Config-driven DROL fusion POD batch analysis")
    parser.add_argument("--config", type=Path,
                        default=Path(r"D:\csu\MyTools\configs\DROL_FusionBatchAnalysis.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    paths = {key: Path(value) for key, value in config["paths"].items()}
    output = paths["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    residuals: dict[str, dict] = defaultdict(dict)
    orbit_epochs: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    satellite_counts: list[dict] = []
    for solution in config["solutions"]:
        name, batch, prefix = solution["name"], solution["batch"], solution["prefix"]
        for date_text in dates(config):
            config_name = f"{prefix}{date_text}"
            sp3 = paths["outputdata_root"] / batch / f"{config_name}.sp3"
            residual = paths["log_root"] / batch / config_name / f"DRO_RDOD_Res_{date_text}.dat"
            editing = paths["tempdata_root"] / batch / config_name / f"DRO_DataEditing_Obs_{date_text}.rnx"
            tag = f"{datetime.strptime(date_text, '%Y%m%d').year}{datetime.strptime(date_text, '%Y%m%d').timetuple().tm_yday:03d}0"
            raw_pattern = solution.get("raw_rinex_pattern", config["raw_rinex_pattern"])
            raw = Path(raw_pattern.format(method=name, date=date_text, tag=tag))
            row = {"solution": name, "label": solution.get("label", name), "date": date_text,
                   "sp3": str(sp3), "raw_rinex": str(raw), "status": "ok"}
            missing = [str(path) for path in (sp3, residual, editing) if not path.exists()]
            if missing:
                row.update({"status": "missing", "missing": missing})
                rows.append(row); continue
            editing_counts = rinex_epoch_counts(editing)
            row.update(residual_metrics(residual))
            row.update({"editing_epochs": len(editing_counts),
                        "editing_observations": sum(count for _, count in editing_counts),
                        "editing_mean_sats": float(np.mean([count for _, count in editing_counts])) if editing_counts else None,
                        "editing_epochs_ge5": sum(count >= 5 for _, count in editing_counts)})
            if raw.exists():
                raw_counts = rinex_epoch_counts(raw)
                row.update({"raw_epochs": len(raw_counts), "raw_observations": sum(count for _, count in raw_counts),
                            "raw_mean_sats": float(np.mean([count for _, count in raw_counts])) if raw_counts else None})
            else:
                raw_counts = []
                row["raw_missing"] = True
            for stage, counts in (("raw_fusion", raw_counts), ("dataediting", editing_counts)):
                satellite_counts.extend({"solution": name, "label": row["label"], "date": date_text,
                                         "stage": stage, "epoch": epoch,
                                         "epoch_utc": epoch.isoformat(sep=" "), "satellite_count": count}
                                        for epoch, count in counts)
            residuals[name].update(analysis.ResLog(str(residual.parent), "DRO", date_text, "RDOD"))
            reference = pre_path(date_text, paths["pre_root"])
            if reference:
                metrics, errors = orbit_comparison(sp3, reference)
                row.update(metrics); row["pre_file"] = str(reference)
                orbit_epochs[name].extend(errors)
            else:
                row.update({"pre_common_epochs": 0, "pre_3d_rms_m": None, "pre_3d_p95_m": None,
                            "pre_r_rms_m": None, "pre_t_rms_m": None, "pre_n_rms_m": None})
            rows.append(row)
    aggregate = summarize(rows, config["quality"]["min_editing_epochs"])
    payload = {"config": config, "records": rows, "aggregate": aggregate}
    (output / "pod_batch_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    clip_ymax_cm = float(config["plots"].get("pre_epoch_rms_ymax_cm", 100.0))
    write_orbit_epoch_csv(orbit_epochs, config["solutions"], output, clip_ymax_cm)
    phase_clip_mm = float(config["plots"].get("posterior_phase_residual_ymax_mm", 60.0))
    write_residual_csv(residuals, config["solutions"], output, phase_clip_mm)
    write_satellite_count_csv(satellite_counts, config["solutions"], output)
    if config["plots"].get("enabled", True):
        plt.rcParams.update({"font.family": "Arial", "figure.dpi": 160})
        plot_results(rows, orbit_epochs, config["solutions"], output, config["plots"].get("dpi", 300),
                     clip_ymax_cm)
        if config["plots"].get("satellite_counts_multiday", True):
            plot_satellite_counts(satellite_counts, config["solutions"], output,
                                  config["plots"].get("dpi", 300))
        if config["plots"].get("residual_multiday", True):
            plot_multiday_residuals(residuals, config["solutions"], output,
                                    config["plots"].get("dpi", 300),
                                    config["plots"].get("posterior_residual", {}))
            if config["plots"].get("posterior_phase_epoch_scatter", False):
                plot_posterior_phase_residual_scatter(residuals, config["solutions"], output,
                                                      config["plots"].get("dpi", 300), phase_clip_mm)
    print(f"Wrote {len(rows)} records to {output / 'pod_batch_summary.json'}")
    for row in aggregate:
        pre = row["pre_3d_rms_median_m"]
        phase = row["phase_rms_median_m"]
        print(f"{row['solution']}: PRE median={pre * 100:.2f} cm" if pre is not None else f"{row['solution']}: PRE unavailable",
              f"phase median={phase * 1000:.2f} mm" if phase is not None else "phase unavailable")


if __name__ == "__main__":
    main()
