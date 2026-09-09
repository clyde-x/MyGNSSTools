"""Polished time-series and sky-view plots for POD posterior residuals."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResidualPlotResult:
    """Files and full-data statistics written for one residual data set."""

    time_series_path: Path
    sky_plot_path: Path
    statistics: pd.DataFrame


PathLike = str | Path


def _parse_timestamp(value: str) -> datetime:
    """Parse the fixed-width CSU residual timestamp without project imports."""
    fields = re.findall(r"\d+(?:\.\d+)?", value)
    if len(fields) < 6:
        raise ValueError(f"Invalid residual timestamp: {value!r}")
    return datetime(*map(int, fields[:5])) + timedelta(seconds=float(fields[5]))


def parse_residual_file(path: PathLike) -> dict[datetime, dict[str, dict[str, float]]]:
    """Read one CSU ``*_Res_*.dat`` file into the plotting data structure.

    The parser intentionally knows only the residual-file record layout:
    timestamp, satellite count, and repeated ``sat/ele/azi/phase/code`` fields.
    """
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"Residual file not found: {file}")
    result: dict[datetime, dict[str, dict[str, float]]] = {}
    for line_number, line in enumerate(file.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            time = _parse_timestamp(line[:26])
            satellite_count = int(line[42:44])
            fields = line[44:].split()
            if len(fields) < satellite_count * 5:
                raise ValueError("truncated satellite fields")
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Invalid residual record in {file} at line {line_number}: {exc}") from exc
        satellites = result.setdefault(time, {})
        for index in range(satellite_count):
            offset = index * 5
            satellite = fields[offset]
            satellites[satellite] = {"ele": float(fields[offset + 1]), "azi": float(fields[offset + 2]),
                                     "prefitL": float(fields[offset + 3]), "prefitC": float(fields[offset + 4])}
    return result


def expand_residual_files(source: PathLike | Sequence[PathLike], *, pattern: str = "*Res*.dat",
                          recursive: bool = True) -> list[Path]:
    """Resolve individual residual files and/or folders to sorted file paths."""
    entries = [source] if isinstance(source, (str, Path)) else list(source)
    if not entries:
        raise ValueError("Residual input cannot be empty.")
    files: list[Path] = []
    for entry in entries:
        path = Path(entry)
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            iterator = path.rglob(pattern) if recursive else path.glob(pattern)
            files.extend(item for item in iterator if item.is_file())
        else:
            raise FileNotFoundError(f"Residual file or folder not found: {path}")
    files = sorted(set(files), key=lambda item: str(item).casefold())
    if not files:
        raise ValueError(f"No residual files matching {pattern!r} found in {', '.join(map(str, entries))}")
    return files


def load_residual_files(source: PathLike | Sequence[PathLike], *, pattern: str = "*Res*.dat",
                        recursive: bool = True) -> dict[datetime, dict[str, dict[str, float]]]:
    """Load and merge all matching daily residual files for one data set."""
    merged: dict[datetime, dict[str, dict[str, float]]] = {}
    for path in expand_residual_files(source, pattern=pattern, recursive=recursive):
        for epoch, satellites in parse_residual_file(path).items():
            merged.setdefault(epoch, {}).update(satellites)
    if not merged:
        raise ValueError("No residual records were found in the selected input files.")
    return merged


def _resolve_source(source: Any, base_dir: Path) -> str | list[str]:
    if isinstance(source, Mapping):
        source = source.get("files", source.get("folder", source.get("path")))
    if source is None:
        raise ValueError("Dataset requires 'inputs', 'files', or 'folder'.")
    values = [source] if isinstance(source, str) else source
    if not isinstance(values, Sequence) or isinstance(values, (bytes, bytearray)):
        raise TypeError("Residual input must be a path string or a list of path strings.")
    resolved = [str(path if Path(path).is_absolute() else base_dir / path) for path in values]
    return resolved[0] if isinstance(source, str) else resolved


def plot_posterior_residuals_from_config(config_path: PathLike) -> list[ResidualPlotResult]:
    """Create residual figures from a dedicated JSON config with no POD paths."""
    file = Path(config_path)
    if not file.is_file():
        raise FileNotFoundError(f"Residual plot config not found: {file}")
    config = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise TypeError("Residual plot config root must be a JSON object.")
    base_dir = file.resolve().parent
    output_value = config.get("output_dir")
    if output_value is None:
        raise ValueError("Residual plot config requires 'output_dir'.")
    output_dir = Path(output_value)
    output_dir = output_dir if output_dir.is_absolute() else base_dir / output_dir
    plot_options = dict(config.get("plot", {}))
    datasets = config.get("datasets")
    if datasets is None:
        datasets = [{"name": config.get("name", "residual"), "inputs": config.get("inputs")}]
    if not isinstance(datasets, Sequence) or isinstance(datasets, (str, bytes)):
        raise TypeError("'datasets' must be a list of dataset objects.")
    results: list[ResidualPlotResult] = []
    comparison_data: dict[str, Mapping[Any, Mapping[str, Mapping[str, Any]]]] = {}
    for index, dataset in enumerate(datasets, start=1):
        if not isinstance(dataset, Mapping):
            raise TypeError("Each dataset must be an object.")
        name = str(dataset.get("name", f"residual_{index}"))
        source = _resolve_source(dataset.get("inputs", dataset), base_dir)
        data = load_residual_files(source, pattern=str(dataset.get("pattern", config.get("pattern", "*Res*.dat"))),
                                   recursive=bool(dataset.get("recursive", config.get("recursive", True))))
        options = {**plot_options, **dict(dataset.get("plot", {}))}
        comparison_data[name] = data
        if not config.get("combine_datasets", False):
            results.append(plot_posterior_residuals(data, name, output_dir, **options))
    if config.get("combine_datasets", False):
        comparison_path = plot_posterior_residual_comparison(
            comparison_data, output_dir, dpi=int(plot_options.get("dpi", 300)),
            scatter_size=float(plot_options.get("scatter_size", 2.0)), alpha=float(plot_options.get("alpha", .35)),
            time_clip_percent=float(plot_options.get("time_clip_percent", 0.0)),
            title=str(config.get("title", "Posterior residual comparison")),
        )
        print(f"Saved combined residual time comparison: {comparison_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot posterior residual time-series and sky-view figures.")
    parser.add_argument("--config", type=Path, required=True, help="Dedicated posterior-residual JSON config.")
    args = parser.parse_args()
    for result in plot_posterior_residuals_from_config(args.config):
        print(result.statistics.to_string(index=False))
        print(f"Saved: {result.time_series_path}\nSaved: {result.sky_plot_path}")


def _to_frame(data: Mapping[Any, Mapping[str, Mapping[str, Any]]], *,
              phase_key: str, code_key: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for epoch, satellites in data.items():
        time = getattr(epoch, "datetime", epoch)
        for satellite, values in satellites.items():
            rows.append({"time": pd.Timestamp(time), "satellite": str(satellite),
                         "elevation": values.get("ele"), "azimuth": values.get("azi"),
                         "phase_m": values.get(phase_key), "code_m": values.get(code_key)})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Residual data contains no observation records.")
    for column in ("elevation", "azimuth", "phase_m", "code_m"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("time")


def _statistics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, label in (("phase_m", "Carrier phase"), ("code_m", "Pseudorange")):
        values = frame[key].dropna().to_numpy()
        if not len(values):
            continue
        rows.append({"observable": label, "records": len(values), "mean_mm": float(values.mean() * 1000),
                     "std_mm": float(values.std() * 1000), "rmse_mm": float(np.sqrt(np.mean(values ** 2)) * 1000),
                     "p95_abs_mm": float(np.quantile(np.abs(values), .95) * 1000)})
    return pd.DataFrame(rows)


def _annotation(statistics: pd.DataFrame, observable: str) -> str:
    row = statistics[statistics.observable.eq(observable)].iloc[0]
    return f"N = {int(row.records):,}\nSTD = {row.std_mm:.2f} mm\nRMSE = {row.rmse_mm:.2f} mm"


def _visible(frame: pd.DataFrame, column: str, clip_percent: float) -> pd.DataFrame:
    subset = frame[np.isfinite(frame[column])].copy()
    if clip_percent == 0 or subset.empty:
        return subset
    limit = subset[column].abs().quantile(1 - clip_percent / 100)
    return subset[subset[column].abs() <= limit]


def plot_posterior_residuals(
    data: Mapping[Any, Mapping[str, Mapping[str, Any]]],
    name: str,
    output_dir: str | Path,
    *,
    phase_key: str = "prefitL",
    code_key: str = "prefitC",
    dpi: int = 300,
    scatter_size: float = 3.0,
    alpha: float = .45,
    time_clip_percent: float = 0.0,
    sky_color_percentile: float = 95.0,
    cmap: str = "coolwarm",
) -> ResidualPlotResult:
    """Write one time-scatter figure and one north-up sky figure.

    ``time_clip_percent`` hides the largest absolute residuals separately for
    phase and code in the time display only. Statistics and sky plots always
    use the complete data. The output consists of ``{name}_posterior_time.png`` and
    ``{name}_posterior_sky.png``.
    """
    if scatter_size <= 0:
        raise ValueError("scatter_size must be positive.")
    if not 0 <= time_clip_percent < 100:
        raise ValueError("time_clip_percent must be in [0, 100).")
    if not 0 < sky_color_percentile <= 100:
        raise ValueError("sky_color_percentile must be in (0, 100].")
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    frame = _to_frame(data, phase_key=phase_key, code_key=code_key)
    statistics = _statistics(frame)
    if len(statistics) != 2:
        raise ValueError(f"{name}: both {phase_key!r} and {code_key!r} need at least one numeric residual.")

    satellite_ids = sorted(frame.satellite.unique())
    colours = dict(zip(satellite_ids, plt.cm.tab20(np.linspace(0, 1, max(1, len(satellite_ids))))))
    fig, axes = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True, layout="constrained")
    for axis, column, label in zip(axes, ("phase_m", "code_m"), ("Carrier-phase posterior residual", "Pseudorange posterior residual")):
        shown = _visible(frame, column, time_clip_percent)
        for satellite, group in shown.groupby("satellite", sort=True):
            axis.scatter(group.time, group[column] * 1000, s=scatter_size, alpha=alpha,
                         color=colours[satellite], linewidths=0, rasterized=True,
                         label=satellite if len(satellite_ids) <= 12 else None)
        axis.axhline(0, color="0.25", linewidth=.8); axis.grid(alpha=.25)
        axis.set_ylabel("Residual (mm)"); axis.set_title(f"{name}: {label}")
        axis.text(.012, .94, _annotation(statistics, "Carrier phase" if column == "phase_m" else "Pseudorange"),
                  transform=axis.transAxes, va="top", fontsize=9,
                  bbox=dict(boxstyle="round", facecolor="white", edgecolor=".55", alpha=.9))
    if len(satellite_ids) <= 12:
        axes[0].legend(title="Satellite", ncol=min(6, len(satellite_ids)), loc="upper right", fontsize=8)
    if time_clip_percent:
        fig.text(.995, .995, f"Time display hides the largest {time_clip_percent:g}% per observable; metrics use all records.",
                 ha="right", va="top", fontsize=8)
    axes[-1].set_xlabel("Time")
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    axes[-1].xaxis.set_major_locator(locator); axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    time_path = output / f"{name}_posterior_time.png"; fig.savefig(time_path, dpi=dpi, bbox_inches="tight"); plt.close(fig)

    fig = plt.figure(figsize=(13, 6.8), layout="constrained")
    for index, (column, label) in enumerate((("phase_m", "Carrier-phase posterior residual"),
                                              ("code_m", "Pseudorange posterior residual")), start=1):
        axis = fig.add_subplot(1, 2, index, projection="polar")
        valid = frame[np.isfinite(frame.azimuth) & np.isfinite(frame.elevation) & np.isfinite(frame[column])]
        values_mm = valid[column].to_numpy() * 1000
        limit = max(float(np.percentile(np.abs(values_mm), sky_color_percentile)), 1e-6)
        points = axis.scatter(np.deg2rad(np.mod(valid.azimuth, 360)), np.clip(90 - valid.elevation, 0, 90),
                              c=values_mm, s=scatter_size, cmap=cmap, norm=Normalize(-limit, limit),
                              alpha=alpha + (1 - alpha) * .35, linewidths=0, rasterized=True)
        axis.set_theta_zero_location("N"); axis.set_theta_direction(-1); axis.set_rlim(0, 90)
        axis.set_rgrids((30, 60, 90), labels=("60°", "30°", "0°"), angle=22.5)
        axis.set_title(f"{name}: {label}", pad=17)
        colorbar = fig.colorbar(points, ax=axis, pad=.1, shrink=.85); colorbar.set_label("O-C residual (mm)")
    sky_path = output / f"{name}_posterior_sky.png"; fig.savefig(sky_path, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    return ResidualPlotResult(time_path, sky_path, statistics)


def plot_posterior_residual_comparison(datasets: Mapping[str, Mapping[Any, Mapping[str, Mapping[str, Any]]]],
                                       output_dir: PathLike, *, dpi: int = 300, scatter_size: float = 2.0,
                                       alpha: float = .35, time_clip_percent: float = 0.0,
                                       title: str = "Posterior residual comparison") -> Path:
    """Overlay all datasets' phase/code time scatter in one figure for comparison."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True, layout="constrained")
    colours = plt.cm.tab10(np.linspace(0, 1, max(1, len(datasets))))
    for (name, data), color in zip(datasets.items(), colours):
        frame = _to_frame(data, phase_key="prefitL", code_key="prefitC")
        for axis, column in zip(axes, ("phase_m", "code_m")):
            shown = _visible(frame, column, time_clip_percent)
            axis.scatter(shown.time, shown[column] * 1000, s=scatter_size, alpha=alpha, color=color,
                         linewidths=0, rasterized=True, label=name)
    for axis, label in zip(axes, ("Carrier-phase posterior residual (mm)", "Pseudorange posterior residual (mm)")):
        axis.axhline(0, color="0.25", linewidth=.8); axis.set_ylabel(label); axis.grid(alpha=.25); axis.legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("Time"); fig.suptitle(title)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8); axes[-1].xaxis.set_major_locator(locator); axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    path = output / "posterior_residual_time_comparison.png"; fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    return path


if __name__ == "__main__":
    main()
