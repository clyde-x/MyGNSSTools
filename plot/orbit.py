"""Configurable multi-product SP3 orbit-comparison figures.

The public entry point deliberately accepts *lists* of SP3 files.  A daily
arc can therefore be split across products without callers concatenating data
or creating intermediate files first.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_LIB_DIR = _ROOT / "lib"
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))
import Sp3Tools


PathLike = str | Path
# A source may be a single SP3 file, a folder containing SP3 files, or an
# explicit collection mixing the two forms.
Sp3Input = PathLike | Sequence[PathLike]


@dataclass(frozen=True)
class OrbitComparisonResult:
    """Data and metrics produced by :func:`plot_sp3_orbit_comparison`."""

    figure: Any
    axes: tuple[Any, ...]
    errors: Mapping[str, pd.DataFrame]
    metrics: pd.DataFrame
    output_path: Path | None


def _resolve_config_source(source: Any, base_dir: Path) -> Sp3Input:
    """Resolve one config-relative SP3 source while retaining its shape."""
    if isinstance(source, Mapping):
        source = source.get("files", source.get("folder", source.get("path")))
    if source is None:
        raise ValueError("SP3 source must provide a file, folder, or 'files' field.")
    values = [source] if isinstance(source, str) else source
    if not isinstance(values, Sequence) or isinstance(values, (bytes, bytearray)):
        raise TypeError("SP3 source must be a path string or a list of path strings.")
    resolved = [str(path if Path(path).is_absolute() else base_dir / path) for path in values]
    return resolved[0] if isinstance(source, str) else resolved


def _resolve_output_path(path: str | Path | None, base_dir: Path) -> Path | None:
    if path is None:
        return None
    value = Path(path)
    return value if value.is_absolute() else base_dir / value


def _as_datetime(value: Any) -> datetime:
    """Convert project ``TimeSystem`` or datetime-like values for plotting."""
    value = getattr(value, "datetime", value)
    return pd.Timestamp(value).to_pydatetime()


def _expand_sp3_files(source: Sp3Input) -> list[Path]:
    """Resolve file/folder input to a sorted, de-duplicated SP3 file list."""
    entries = [source] if isinstance(source, (str, Path)) else list(source)
    if not entries:
        raise ValueError("SP3 input cannot be empty.")
    files: list[Path] = []
    for entry in entries:
        path = Path(entry)
        if path.is_file():
            if path.suffix.lower() != ".sp3":
                raise ValueError(f"Expected an .sp3 file, got: {path}")
            files.append(path)
        elif path.is_dir():
            # Folder input includes all nested SP3 arcs/products, so callers
            # can point at a day/batch directory without pre-building a list.
            files.extend(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() == ".sp3")
        else:
            raise FileNotFoundError(f"SP3 file or folder not found: {path}")
    files = sorted(set(files), key=lambda item: str(item).casefold())
    if not files:
        raise ValueError(f"No .sp3 files found in: {', '.join(map(str, entries))}")
    return files


def _read_sp3(source: Sp3Input, *, start_time: Any = None, end_time: Any = None) -> pd.DataFrame:
    files = _expand_sp3_files(source)
    frames: list[pd.DataFrame] = []
    for file in files:
        frame = Sp3Tools.SP3(str(file)).asDataFrame()
        if frame.empty:
            print(f"Warning: No position records read from {file}.", file=sys.stderr)
            continue
        frames.append(frame.loc[:, ["timesystem", "sat_id", "x", "y", "z"]].copy())
    if not frames:
        raise ValueError(f"No position records read from: {list(map(str, files))}")
    frame = pd.concat(frames, ignore_index=True)
    frame["time"] = frame["timesystem"].map(_as_datetime)
    if start_time is not None:
        frame = frame[frame.time >= _as_datetime(start_time)]
    if end_time is not None:
        frame = frame[frame.time <= _as_datetime(end_time)]
    # A product list may overlap at a boundary. Identical epoch/satellite rows
    # are safe; keeping the first one gives deterministic behaviour.
    return frame.drop_duplicates(["time", "sat_id"], keep="first").sort_values(["time", "sat_id"])


def _normalise_comparisons(comparisons: Mapping[str, Sp3Input] | Sequence[Any]) -> list[tuple[str, Sp3Input]]:
    if isinstance(comparisons, Mapping):
        return [(str(name), files) for name, files in comparisons.items()]
    result: list[tuple[str, Sp3Input]] = []
    for index, item in enumerate(comparisons, start=1):
        if isinstance(item, tuple) and len(item) == 2:
            first, second = item
            # Both ``(files, name)`` and ``(name, files)`` are intentionally
            # accepted because existing scripts use both conventions.  If both
            # values are strings, an existing path is treated as the source;
            # this makes ``(r"folder", "Name")`` work naturally.
            if isinstance(first, Path):
                files, name = first, second
            elif isinstance(second, Path):
                name, files = first, second
            elif isinstance(first, str) and isinstance(second, str) and Path(first).exists():
                files, name = first, second
            elif isinstance(first, str) and isinstance(second, str) and Path(second).exists():
                name, files = first, second
            elif isinstance(second, str):
                files, name = first, second
            elif isinstance(first, str):
                name, files = first, second
            else:
                raise TypeError("Named comparison must be (files, name) or (name, files).")
        else:
            files, name = item, f"Comparison {index}"
        result.append((str(name), files))
    if not result:
        raise ValueError("At least one comparison SP3 list is required.")
    return result


def _select_satellite(frame: pd.DataFrame, label: str, selected: str | None) -> str:
    """Choose one satellite from one product group, independently of its ID."""
    available = sorted(frame.sat_id.unique())
    if selected is not None:
        if selected not in available:
            raise ValueError(f"Satellite {selected!r} was requested for {label!r}, but available IDs are {available}.")
        return selected
    if len(available) == 1:
        return available[0]
    raise ValueError(f"{label!r} contains multiple satellites {available}; specify satellite as a string or a mapping.")


def plot_sp3_orbit_comparison(
    reference_files: Sp3Input,
    comparisons: Mapping[str, Sp3Input] | Sequence[Any],
    *,
    satellite: str | Mapping[str, str] | None = None,
    reference_label: str = "Reference",
    title: str = "SP3 orbit comparison",
    kind: str = "plot",
    view: str = "xyz",
    scatter_size: float = 3.0,
    styles: Mapping[str, Mapping[str, Any]] | None = None,
    start_time: Any = None,
    end_time: Any = None,
    output_path: PathLike | None = None,
    figsize: tuple[float, float] = (13, 9),
    dpi: int = 300,
    unit: str = "mm",
    show_rmse: bool = True,
    trim_percent: float = 0.0,
    grid: bool = True,
    axes: Sequence[Any] | None = None,
) -> OrbitComparisonResult:
    """Compare multiple SP3-product lists with one reference product list.

    Parameters
    ----------
    reference_files:
        One or more reference SP3 files, or a folder.  A folder is searched
        recursively for every ``.sp3`` file (case-insensitive).
    comparisons:
        Either ``{"solution name": files_or_folder}``, raw file/folder
        inputs, or named tuples ``(files_or_folder, "solution name")`` /
        reversed tuples.  A folder is searched recursively for ``.sp3`` files.
    satellite:
        Satellite ID to compare, or a mapping when the reference and tested
        products use different IDs.  Omit it when *each input group* contains
        only one satellite: each group is then selected independently.  For
        example a reference ``L98`` product can be compared to an ``L02``
        product without any satellite setting.  A mapping uses ``reference``
        and comparison names, e.g. ``{"reference": "L98", "Final": "L02"}``.
    kind:
        ``"plot"`` for line charts or ``"scatter"`` for scatter charts.
    view:
        ``"xyz"`` draws separate X/Y/Z error axes, ``"3d"`` draws only the
        3D position-error magnitude, and ``"both"`` draws all four axes.
    scatter_size:
        Default marker area for ``kind="scatter"``. A per-comparison
        ``styles[name]["s"]`` value takes precedence.
    styles:
        Per-comparison Matplotlib keyword dictionaries keyed by solution name;
        e.g. ``{"Final": {"color": "#0072B2", "marker": "o",
        "linestyle": "--", "lw": 1.2}}``. In scatter mode use normal
        scatter options such as ``s`` and ``alpha``.
    trim_percent:
        Percentage (0 to less than 100) of the largest *3D-error* samples to
        hide from each plotted solution.  It affects display only; RMSE and
        the returned ``errors`` tables always retain every common epoch.

    Returns a result object containing the figure, per-solution error tables,
    RMSE metrics, and the saved path (when ``output_path`` is supplied).
    Position fields in this project are SP3 kilometres; differences are
    converted to the requested ``unit`` (``mm``, ``m``, or ``km``).
    """
    if kind not in {"plot", "scatter"}:
        raise ValueError("kind must be 'plot' or 'scatter'.")
    if view not in {"xyz", "3d", "both"}:
        raise ValueError("view must be 'xyz', '3d', or 'both'.")
    if not 0 <= trim_percent < 100:
        raise ValueError("trim_percent must be in the range [0, 100).")
    if scatter_size <= 0:
        raise ValueError("scatter_size must be positive.")
    scale = {"km": 1.0, "m": 1_000.0, "mm": 1_000_000.0}.get(unit)
    if scale is None:
        raise ValueError("unit must be one of: 'km', 'm', 'mm'.")

    named = _normalise_comparisons(comparisons)
    reference = _read_sp3(reference_files, start_time=start_time, end_time=end_time)
    tested = [(name, _read_sp3(files, start_time=start_time, end_time=end_time)) for name, files in named]
    if satellite is not None and not isinstance(satellite, (str, Mapping)):
        raise TypeError("satellite must be a string, a mapping, or None.")
    satellite_ids = satellite if isinstance(satellite, Mapping) else {}
    reference_satellite = _select_satellite(reference, reference_label,
                                            satellite_ids.get("reference") if satellite_ids else satellite)
    reference = reference[reference.sat_id.eq(reference_satellite)]
    if reference.empty:
        raise ValueError(f"Reference products contain no records for satellite {reference_satellite!r}.")

    errors: dict[str, pd.DataFrame] = {}
    metric_rows: list[dict[str, Any]] = []
    for name, frame in tested:
        tested_satellite = _select_satellite(frame, name, satellite_ids.get(name) if satellite_ids else satellite)
        frame = frame[frame.sat_id.eq(tested_satellite)]
        # Satellite labels are product identifiers, not necessarily identical
        # across agencies (e.g. L98 in PRE versus L02 in POD).  Alignment is
        # therefore based on epoch after each product selects its one target.
        merged = reference.merge(frame, on="time", how="inner", suffixes=("_ref", "_test"))
        if merged.empty:
            raise ValueError(f"No common epochs between {reference_label!r} ({reference_satellite}) and "
                             f"{name!r} ({tested_satellite}).")
        error = pd.DataFrame({"time": merged.time})
        for component in ("x", "y", "z"):
            error[component] = (merged[f"{component}_ref"] - merged[f"{component}_test"]) * scale
        error["3d"] = np.linalg.norm(error[["x", "y", "z"]].to_numpy(), axis=1)
        errors[name] = error
        metric_rows.append({"name": name, "reference_satellite": reference_satellite,
                            "tested_satellite": tested_satellite, "epochs": len(error),
                            "x_rmse": float(np.sqrt(np.mean(error.x ** 2))),
                            "y_rmse": float(np.sqrt(np.mean(error.y ** 2))),
                            "z_rmse": float(np.sqrt(np.mean(error.z ** 2))),
                            "3d_rmse": float(np.sqrt(np.mean(error["3d"] ** 2)))})
    displayed: dict[str, pd.DataFrame] = {}
    plotted: dict[str, pd.DataFrame] = {}
    for row in metric_rows:
        name, error = row["name"], errors[row["name"]]
        threshold = float(error["3d"].quantile(1 - trim_percent / 100)) if trim_percent else float("inf")
        displayed[name] = error[error["3d"] <= threshold]
        # Keep the full time coordinate but mask rejected values.  Matplotlib
        # then breaks a line at an outlier instead of joining its two sides.
        plotted[name] = error.copy()
        plotted[name].loc[plotted[name]["3d"] > threshold, ["x", "y", "z", "3d"]] = np.nan
        row["displayed_epochs"] = len(displayed[name])
        row["trim_percent"] = trim_percent
        row["trim_3d_threshold"] = threshold if trim_percent else np.nan
    metrics = pd.DataFrame(metric_rows)

    if axes is None:
        axis_count = {"xyz": 3, "3d": 1, "both": 4}[view]
        figure, axes_array = plt.subplots(axis_count, 1, figsize=figsize, sharex=True, layout="constrained")
        axes_array = np.atleast_1d(axes_array)
    else:
        axes_array = np.asarray(axes)
        expected = {"xyz": 3, "3d": 1, "both": 4}[view]
        if len(axes_array) != expected:
            raise ValueError(f"axes must contain {expected} Matplotlib axes for view={view!r}.")
        figure = axes_array[0].figure
    xyz_axes = axes_array[:3] if view in {"xyz", "both"} else ()
    norm_axis = axes_array[-1] if view in {"3d", "both"} else None
    styles = styles or {}
    defaults = [{"color": color, "lw": 1.0} for color in ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9")]
    for index, (name, error) in enumerate(plotted.items()):
        options = {**defaults[index % len(defaults)], **styles.get(name, {})}
        for axis, component in zip(xyz_axes, ("x", "y", "z")):
            if kind == "plot":
                axis.plot(error.time, error[component], label=name, **options)
            else:
                options.setdefault("s", scatter_size)
                axis.scatter(error.time, error[component], label=name, **options)
        if norm_axis is not None:
            if kind == "plot":
                norm_axis.plot(error.time, error["3d"], label=name, **options)
            else:
                options.setdefault("s", scatter_size)
                norm_axis.scatter(error.time, error["3d"], label=name, **options)
    for axis, component in zip(xyz_axes, ("X", "Y", "Z")):
        axis.set_ylabel(f"{component} error ({unit})")
        if grid:
            axis.grid(True, alpha=.3)
    if norm_axis is not None:
        norm_axis.set_ylabel(f"3D error ({unit})")
        if grid:
            norm_axis.grid(True, alpha=.3)
    axes_array[0].legend(loc="upper right")
    axes_array[-1].set_xlabel("Time")
    figure.suptitle(title)
    if show_rmse:
        text = "\n".join(f"{row['name']}: 3D RMSE = {row['3d_rmse']:.3f} {unit}" for row in metric_rows)
        if trim_percent:
            text += f"\nDisplay: largest {trim_percent:g}% 3D errors removed"
        axes_array[0].text(.015, .96, text, transform=axes_array[0].transAxes, va="top",
                           bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": ".5", "alpha": .9})
    saved = Path(output_path) if output_path is not None else None
    if saved is not None:
        saved.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(saved, dpi=dpi, bbox_inches="tight")
    return OrbitComparisonResult(figure, tuple(axes_array), errors, metrics, saved)


def plot_sp3_orbit_comparison_from_config(config_path: PathLike) -> OrbitComparisonResult:
    """Run :func:`plot_sp3_orbit_comparison` using a JSON configuration file.

    Relative file and folder paths are resolved relative to the configuration
    file, rather than to the command's current working directory.
    """
    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"Orbit plot config not found: {config_file}")
    try:
        config = json.loads(config_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in orbit plot config {config_file}: {exc}") from exc
    if not isinstance(config, Mapping):
        raise TypeError("Orbit plot config root must be a JSON object.")
    base_dir = config_file.resolve().parent
    reference_source = config.get("reference_files", config.get("reference"))
    if reference_source is None:
        raise ValueError("Orbit plot config requires 'reference_files' (or 'reference').")

    raw_comparisons = config.get("comparisons")
    if not raw_comparisons:
        raise ValueError("Orbit plot config requires at least one item in 'comparisons'.")
    styles = dict(config.get("styles", {}))
    comparisons: dict[str, Sp3Input] = {}
    if isinstance(raw_comparisons, Mapping):
        comparisons = {str(name): _resolve_config_source(source, base_dir)
                       for name, source in raw_comparisons.items()}
    elif isinstance(raw_comparisons, Sequence):
        for index, item in enumerate(raw_comparisons, start=1):
            if not isinstance(item, Mapping):
                raise TypeError("Each comparisons item must be an object with 'files'/'folder' and optional 'name'.")
            name = str(item.get("name", f"Comparison {index}"))
            comparisons[name] = _resolve_config_source(item, base_dir)
            if "style" in item:
                styles[name] = item["style"]
    else:
        raise TypeError("'comparisons' must be an object or a list of objects.")

    allowed = {"satellite", "reference_label", "title", "kind", "view", "scatter_size", "start_time", "end_time", "figsize",
               "dpi", "unit", "show_rmse", "trim_percent", "grid"}
    options = {key: config[key] for key in allowed if key in config}
    if "figsize" in options:
        options["figsize"] = tuple(options["figsize"])
    return plot_sp3_orbit_comparison(
        _resolve_config_source(reference_source, base_dir), comparisons,
        styles=styles or None,
        output_path=_resolve_output_path(config.get("output_path"), base_dir),
        **options,
    )


def main() -> None:
    """Command line interface for a config-driven SP3 orbit comparison."""
    parser = argparse.ArgumentParser(description="Plot multi-product SP3 orbit errors from a JSON configuration.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the orbit-comparison JSON config.")
    args = parser.parse_args()
    result = plot_sp3_orbit_comparison_from_config(args.config)
    print(result.metrics.to_string(index=False))
    if result.output_path:
        print(f"Saved figure: {result.output_path}")


if __name__ == "__main__":
    main()
