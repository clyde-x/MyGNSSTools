"""Configuration-driven RINEX satellite-count time-series plots."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PathLike = str | Path
SYSTEM_LABELS = {"G": "GPS", "C": "BDS", "R": "GLONASS", "E": "Galileo", "J": "QZSS",
                 "S": "SBAS", "I": "NavIC", "ALL": "All systems"}
DEFAULT_COLORS = {"G": "#0072B2", "C": "#E69F00", "R": "#CC79A7", "E": "#009E73",
                  "J": "#D55E00", "S": "#56B4E9", "I": "#999999", "ALL": "#222222"}


def expand_rinex_files(source: PathLike | Sequence[PathLike], *, pattern: str = "*.rnx",
                       recursive: bool = True) -> list[Path]:
    entries = [source] if isinstance(source, (str, Path)) else list(source)
    files: list[Path] = []
    for entry in entries:
        path = Path(entry)
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            finder = path.rglob(pattern) if recursive else path.glob(pattern)
            files.extend(item for item in finder if item.is_file())
        else:
            raise FileNotFoundError(f"RINEX file or folder not found: {path}")
    files = sorted(set(files), key=lambda item: str(item).casefold())
    if not files:
        raise ValueError(f"No RINEX files matching {pattern!r} found.")
    return files


def _parse_v3(lines: list[str]) -> list[tuple[datetime, list[str]]]:
    rows, time, satellites = [], None, []
    for line in lines:
        if line.startswith(">"): 
            if time is not None:
                rows.append((time, satellites))
            values = line[1:].split()
            second = float(values[5])
            time = datetime(*map(int, values[:5])) + timedelta(seconds=second)
            satellites = []
        elif time is not None and len(line) >= 3 and line[0].isalpha() and line[1:3].isdigit():
            satellites.append(line[:3])
    if time is not None:
        rows.append((time, satellites))
    return rows


def _parse_v2(lines: list[str]) -> list[tuple[datetime, list[str]]]:
    rows, index = [], 0
    while index < len(lines):
        line = lines[index]
        try:
            year = int(line[0:3]); year += 2000 if year < 80 else 1900
            second = float(line[15:26]); flag = int(line[28:29]); count = int(line[29:32])
        except ValueError:
            index += 1; continue
        if flag > 1:
            index += 1; continue
        time = datetime(year, int(line[3:6]), int(line[6:9]), int(line[9:12]), int(line[12:15])) + timedelta(seconds=second)
        payload = line[32:68]; satellites = [payload[pos:pos + 3].strip() for pos in range(0, len(payload), 3) if payload[pos:pos + 3].strip()]
        while len(satellites) < count and index + 1 < len(lines):
            index += 1; payload = lines[index][:68]
            satellites.extend(payload[pos:pos + 3].strip() for pos in range(0, len(payload), 3) if payload[pos:pos + 3].strip())
        rows.append((time, satellites[:count])); index += 1
    return rows


def read_rinex_satellite_counts(files: Sequence[PathLike]) -> pd.DataFrame:
    """Read RINEX 2/3 epochs and count unique observed satellites by system."""
    rows: list[dict[str, Any]] = []
    for file in files:
        path = Path(file); lines = path.read_text(encoding="ascii", errors="ignore").splitlines()
        try:
            header_end = next(index for index, line in enumerate(lines) if "END OF HEADER" in line)
        except StopIteration as exc:
            raise ValueError(f"RINEX header terminator not found: {path}") from exc
        version = float(lines[0][:9]) if lines and lines[0][:9].strip() else 3.0
        epochs = _parse_v3(lines[header_end + 1:]) if version >= 3 else _parse_v2(lines[header_end + 1:])
        for time, satellites in epochs:
            systems = {satellite[0] for satellite in satellites if satellite}
            row = {"time": time, "file": path.name, "ALL": len(set(satellites))}
            row.update({system: sum(satellite.startswith(system) for satellite in set(satellites)) for system in systems})
            rows.append(row)
    if not rows:
        raise ValueError("No observation epochs were found in the selected RINEX files.")
    return pd.DataFrame(rows).groupby("time", as_index=False).max().sort_values("time")


def plot_satellite_counts(frame: pd.DataFrame, name: str, output_dir: PathLike, *,
                          systems: Sequence[str] = ("G", "C", "ALL"), title: str | None = None,
                          dpi: int = 300, styles: Mapping[str, Mapping[str, Any]] | None = None) -> tuple[Path, pd.DataFrame]:
    """Plot selected GNSS-system satellite counts and return summary statistics."""
    requested = [str(system).upper() for system in systems]
    available = [system for system in requested if system in frame.columns]
    if not available:
        raise ValueError(f"None of requested systems {requested} occurs in the RINEX data.")
    values = frame.set_index("time").reindex(columns=available, fill_value=0).fillna(0)
    stats = pd.DataFrame({"system": available, "label": [SYSTEM_LABELS.get(system, system) for system in available],
                          "mean": [values[system].mean() for system in available], "min": [values[system].min() for system in available],
                          "max": [values[system].max() for system in available], "epochs": len(values)})
    fig, axis = plt.subplots(figsize=(14, 6.5), layout="constrained"); styles = styles or {}
    for system in available:
        row = stats[stats.system.eq(system)].iloc[0]
        options = {"color": DEFAULT_COLORS.get(system), "lw": 1.1, **styles.get(system, {})}
        axis.plot(values.index, values[system], label=f"{row.label} (μ={row['mean']:.1f}, min={row['min']:.0f}, max={row['max']:.0f})", **options)
    axis.set(title=title or f"{name}: observed satellite count", xlabel="Time", ylabel="Number of satellites")
    axis.grid(alpha=.28); axis.legend(loc="upper right")
    locator = mdates.AutoDateLocator(minticks=4, maxticks=9); axis.xaxis.set_major_locator(locator); axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    image = output / f"{name}_satellite_count.png"; csv = output / f"{name}_satellite_count.csv"
    fig.savefig(image, dpi=dpi, bbox_inches="tight"); plt.close(fig); values.reset_index().to_csv(csv, index=False); stats.to_csv(output / f"{name}_satellite_count_summary.csv", index=False)
    return image, stats


def plot_satellite_count_comparison(frames: Mapping[str, pd.DataFrame], output_dir: PathLike, *,
                                    systems: Sequence[str], title: str = "RINEX satellite-count comparison",
                                    dpi: int = 300, styles: Mapping[str, Mapping[str, Any]] | None = None,
                                    dataset_colors: Mapping[str, str] | None = None) -> Path:
    """Overlay all selected dataset/system count curves in one comparison figure."""
    fig, axis = plt.subplots(figsize=(14, 6.5), layout="constrained"); styles = styles or {}; dataset_colors = dataset_colors or {}
    palette = plt.cm.tab10(np.linspace(0, 1, max(1, len(frames))))
    rows = []
    for data_index, (name, frame) in enumerate(frames.items()):
        requested = [str(system).upper() for system in systems]
        for system_index, system in enumerate(system for system in requested if system in frame.columns):
            values = frame.set_index("time")[system].fillna(0)
            key = f"{name}:{system}"
            options = {"color": dataset_colors.get(name, palette[data_index]), "lw": 1.1,
                       "linestyle": ("-", "--", ":", "-.")[system_index % 4], **styles.get(key, {})}
            axis.plot(values.index, values, label=f"{name} · {SYSTEM_LABELS.get(system, system)}", **options)
            rows.append({"dataset": name, "system": system, "mean": values.mean(), "min": values.min(), "max": values.max()})
    if not rows:
        raise ValueError("No requested GNSS-system data is available for comparison.")
    axis.set(title=title, xlabel="Time", ylabel="Number of satellites"); axis.grid(alpha=.28)
    axis.legend(loc="upper right", ncol=2, fontsize=8)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=9); axis.xaxis.set_major_locator(locator); axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    image = output / "satellite_count_comparison.png"; fig.savefig(image, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(rows).to_csv(output / "satellite_count_comparison_summary.csv", index=False)
    return image


def _resolve_source(source: Any, base: Path) -> str | list[str]:
    if isinstance(source, Mapping): source = source.get("files", source.get("folder", source.get("path")))
    values = [source] if isinstance(source, str) else source
    if not values: raise ValueError("Dataset requires 'inputs', 'files', or 'folder'.")
    resolved = [str(path if Path(path).is_absolute() else base / path) for path in values]
    return resolved[0] if isinstance(source, str) else resolved


def plot_satellite_counts_from_config(config_path: PathLike) -> list[tuple[Path, pd.DataFrame]]:
    file = Path(config_path); config = json.loads(file.read_text(encoding="utf-8")); base = file.resolve().parent
    output = Path(config["output_dir"]); output = output if output.is_absolute() else base / output
    datasets = config.get("datasets", [{"name": config.get("name", "rinex"), "inputs": config.get("inputs")}])
    results, frames = [], {}
    for index, dataset in enumerate(datasets, start=1):
        name = str(dataset.get("name", f"rinex_{index}")); source = _resolve_source(dataset.get("inputs", dataset), base)
        files = expand_rinex_files(source, pattern=str(dataset.get("pattern", config.get("pattern", "*.rnx"))), recursive=bool(dataset.get("recursive", config.get("recursive", True))))
        frame = read_rinex_satellite_counts(files)
        frames[name] = frame
        if not config.get("combine_datasets", False):
            results.append(plot_satellite_counts(frame, name, output, systems=dataset.get("systems", config.get("systems", ["G", "C", "ALL"])),
                                                  title=dataset.get("title", config.get("title")), dpi=int(config.get("dpi", 300)), styles=dataset.get("styles", config.get("styles", {}))))
    if config.get("combine_datasets", False):
        image = plot_satellite_count_comparison(frames, output, systems=config.get("systems", ["G", "C", "ALL"]),
                                                title=config.get("title", "RINEX satellite-count comparison"), dpi=int(config.get("dpi", 300)),
                                                styles=config.get("comparison_styles", {}), dataset_colors=config.get("dataset_colors", {}))
        results.append((image, pd.DataFrame()))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot selected-system satellite counts from RINEX files.")
    parser.add_argument("--config", type=Path, required=True); args = parser.parse_args()
    for image, stats in plot_satellite_counts_from_config(args.config):
        print(stats.to_string(index=False)); print(f"Saved: {image}")


if __name__ == "__main__": main()
