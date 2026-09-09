"""Config-driven DROL dual-antenna fusion entry point.

This is the only public fusion command.  Select one or more methods in the
JSON config; all per-day input/output paths are rendered from its templates.
The legacy policy modules are imported only as implementation libraries.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

# The runnable entry point lives in MyTools; make its shared modules available
# as well when this file is staged or tested from another directory.
MYTOOLS_ROOT = Path(r"D:\csu\MyTools")
for _module_dir in (MYTOOLS_ROOT / "lib", MYTOOLS_ROOT / "src"):
    if str(_module_dir) not in sys.path:
        sys.path.insert(0, str(_module_dir))

import Rinex
import merge_priority
from merge_dual_antenna_geometry import prepare_attitude, prepare_sp3, select_observations
from fuse_rinex_hybrid_by_attitude import attitude_row, carrier_arc_lengths, smooth_minimum_runs


@dataclass(frozen=True)
class Inputs:
    ant1: Path
    ant2: Path
    attitude: Path
    leo_sp3: Path
    gnss_sp3: Path


def days(date_range: dict[str, str]) -> list[datetime]:
    start = datetime.strptime(date_range["start"], "%Y%m%d")
    end = datetime.strptime(date_range["end"], "%Y%m%d")
    if end < start:
        raise ValueError("date_range.end must not be earlier than start")
    return [start + timedelta(days=index) for index in range((end - start).days + 1)]


def render(pattern: str, day: datetime, method: str) -> Path:
    tag = f"{day.year}{day.timetuple().tm_yday:03d}0"
    return Path(pattern.format(date=day.strftime("%Y%m%d"), tag=tag, method=method))


def priority(inputs: Inputs, output: Path, parameters: dict[str, Any]) -> dict[str, int]:
    """Valid-arc union with antenna 2 as the primary source."""
    header, ant2 = merge_priority.read_rinex_obs(str(inputs.ant2), system="MIX")
    _, ant1 = merge_priority.read_rinex_obs(str(inputs.ant1), system="MIX")
    merged, records = merge_priority.merge_rinex_data_with_strategy1(
        ant2, ant1, min_continuous_time=float(parameters.get("minimum_arc_seconds", 50.0))
    )
    # ``merge_priority`` exposes a legacy wrapper whose records contain
    # metadata plus an ``observations`` dictionary.  Convert through its
    # matching writer; feeding that wrapper directly to ``Rinex.writeObs``
    # mistakes the metadata dictionary for satellite observations.
    merge_priority.write_merged_rinex(str(output), header, merged)
    return {"epochs": len(merged), "observations": sum(len(value["observations"]) for value in merged.values()),
            "selection_records": len(records)}


def geometry(inputs: Inputs, output: Path, parameters: dict[str, Any], policy: str) -> dict[str, int]:
    """Geometry, hysteresis, or front-hemisphere screen fusion."""
    header = Rinex.readObsHead(str(inputs.ant1))
    ant1 = Rinex.readObs(str(inputs.ant1), header)
    ant2 = Rinex.readObs(str(inputs.ant2), Rinex.readObsHead(str(inputs.ant2)))
    model = (prepare_sp3([inputs.leo_sp3]), prepare_sp3([inputs.gnss_sp3]), prepare_attitude(inputs.attitude))
    merged, _ = select_observations(
        ant1, ant2, policy, model,
        float(parameters.get("minimum_arc_seconds", 50.0)),
        float(parameters.get("hysteresis", 0.15)),
        float(parameters.get("minimum_score", 0.0)),
    )
    Rinex.writeObs(header, merged, str(output))
    return {"epochs": len(merged), "observations": sum(len(value) for value in merged.values())}


def attitude_hybrid(inputs: Inputs, output: Path, parameters: dict[str, Any], switch_only: bool) -> dict[str, int]:
    """Whole-antenna attitude switch, optionally with side-window arc fusion."""
    header = Rinex.readObsHead(str(inputs.ant1))
    ant1 = Rinex.readObs(str(inputs.ant1), header)
    ant2 = Rinex.readObs(str(inputs.ant2), Rinex.readObsHead(str(inputs.ant2)))
    leo, attitude = prepare_sp3([inputs.leo_sp3]), prepare_attitude(inputs.attitude)
    rows = [attitude_row(epoch, leo, attitude) for epoch in sorted(set(ant1) | set(ant2))]
    rows = [row for row in rows if row is not None]
    valid_epochs = [row["epoch"] for row in rows]
    labels = ["ant2" if row["theta_ant2_deg"] <= row["theta_ant1_deg"] else "ant1" for row in rows]
    labels = smooth_minimum_runs(labels, valid_epochs, float(parameters.get("minimum_switch_minutes", 20.0)) * 60)
    schedule = dict(zip(valid_epochs, labels))
    limit = np.sin(np.deg2rad(float(parameters.get("side_window_deg", 5.0))))
    arcs1, arcs2 = carrier_arc_lengths(ant1), carrier_arc_lengths(ant2)
    merged: dict = {}
    for row in rows:
        epoch, chosen = row["epoch"], schedule[row["epoch"]]
        if switch_only or not abs(row["zenith_dot_ant2"]) <= limit:
            source = ant1 if chosen == "ant1" else ant2
            if epoch in source:
                merged[epoch] = source[epoch].copy()
            continue
        selected = {}
        for satellite in sorted(set(ant1.get(epoch, {})) | set(ant2.get(epoch, {}))):
            one, two = satellite in ant1.get(epoch, {}), satellite in ant2.get(epoch, {})
            if one and two:
                source = ant1 if arcs1[epoch].get(satellite, 0) >= arcs2[epoch].get(satellite, 0) else ant2
            else:
                source = ant1 if one else ant2
            selected[satellite] = source[epoch][satellite]
        if selected:
            merged[epoch] = selected
    Rinex.writeObs(header, merged, str(output))
    return {"epochs": len(merged), "observations": sum(len(value) for value in merged.values())}


def run_method(method: str, inputs: Inputs, output: Path, parameters: dict[str, Any]) -> dict[str, int]:
    """Uniform policy interface used by the configuration-driven runner."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if method == "single_ant1":
        shutil.copyfile(inputs.ant1, output)
        return {"epochs": 0, "observations": 0}
    if method == "single_ant2":
        shutil.copyfile(inputs.ant2, output)
        return {"epochs": 0, "observations": 0}
    if method == "priority":
        return priority(inputs, output, parameters)
    if method in {"geometry", "geometry_hysteresis", "geometry_screen"}:
        return geometry(inputs, output, parameters, method)
    if method == "hybrid":
        return attitude_hybrid(inputs, output, parameters, switch_only=False)
    if method == "attitude_switch":
        return attitude_hybrid(inputs, output, parameters, switch_only=True)
    raise ValueError(f"Unsupported fusion method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DROL fusion driven by JSON configuration")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).with_name("DROL_FusionConfig.example.json"))
    parser.add_argument("--dry-run", action="store_true", help="Validate and print planned work without writing RINEX.")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    required = {"ant1", "ant2", "attitude", "leo_sp3", "gnss_sp3"}
    if set(config["inputs"]) != required:
        raise ValueError(f"inputs must contain exactly: {sorted(required)}")
    for day in days(config["date_range"]):
        base = {key: render(pattern, day, "") for key, pattern in config["inputs"].items()}
        missing = [str(path) for path in base.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing input(s):\n" + "\n".join(missing))
        for task in config["methods"]:
            if not task.get("enabled", True):
                continue
            method = task["method"]
            name = task.get("name", method)
            inputs = Inputs(**base)
            output = render(config["output"]["pattern"], day, name)
            if args.dry_run:
                print(f"[plan] {day:%Y%m%d} {name}: {output}")
                continue
            result = run_method(method, inputs, output, task.get("parameters", {}))
            print(f"[done] {day:%Y%m%d} {name}: epochs={result['epochs']} observations={result['observations']} -> {output}")


if __name__ == "__main__":
    main()
