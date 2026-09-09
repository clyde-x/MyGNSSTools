#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析双天线原始 RINEX 与融合后 RINEX 的观测数量、互补性、SNR 和载波弧段连续性。

目录约定：
    MyTools/
    ├─ lib/Rinex.py
    ├─ src/analyze_rinex_observations.py
    └─ configs/rinex_analysis.json

运行：
    python src/analyze_rinex_observations.py --config configs/rinex_analysis.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIB_DIR = PROJECT_ROOT / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

try:
    from Rinex import readObsHead, readRinex3LineTime
except ImportError as exc:
    raise ImportError(f"无法导入 {LIB_DIR / 'Rinex.py'}，请确认 lib 目录位置正确。") from exc

LOGGER = logging.getLogger("rinex-analysis")


@dataclass(frozen=True)
class FileRecord:
    dataset: str
    antenna: Optional[int]
    year: int
    doy: int
    date: datetime
    path: Path

    @property
    def day_id(self) -> str:
        return f"{self.year}{self.doy:03d}"


@dataclass(frozen=True)
class ObsValue:
    value: Optional[float]
    lli: int = 0
    ssi: int = 0


@dataclass
class DatasetResult:
    record: FileRecord
    header: Dict[str, Any]
    interval_s: float
    epoch_valid_sats: Dict[datetime, Set[str]]
    epoch_recorded_sats: Dict[datetime, Set[str]]
    arcs: List[Dict[str, Any]]
    snr_values: Dict[str, List[float]]
    summary: Dict[str, Any]


class RinexFormatError(RuntimeError):
    pass


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    root_cfg = config.get("project_root")
    root = (path.parent / root_cfg).resolve() if root_cfg else PROJECT_ROOT
    config["_project_root"] = root
    for key in ("raw_dir", "merged_dir", "output_dir"):
        value = config["paths"][key]
        p = Path(value)
        config["paths"][key] = p.resolve() if p.is_absolute() else (root / p).resolve()
    return config


def parse_year_doy(text: str) -> datetime:
    match = re.fullmatch(r"(\d{4})[-/]?(\d{3})", text.strip())
    if not match:
        raise ValueError(f"日期应写成 YYYY-DOY，例如 2025-001；当前值：{text}")
    year, doy = int(match.group(1)), int(match.group(2))
    if doy < 1 or doy > (366 if _is_leap_year(year) else 365):
        raise ValueError(f"非法年积日：{text}")
    return datetime(year, 1, 1) + timedelta(days=doy - 1)


def _is_leap_year(year: int) -> bool:
    return year % 400 == 0 or (year % 4 == 0 and year % 100 != 0)


def date_to_year_doy(date: datetime) -> Tuple[int, int]:
    return date.year, int(date.strftime("%j"))


def discover_files(config: Mapping[str, Any]) -> Dict[Tuple[str, Optional[int], int, int], FileRecord]:
    paths = config["paths"]
    naming = config["naming"]
    start = parse_year_doy(config["date_range"]["start"])
    end = parse_year_doy(config["date_range"]["end"])
    if end < start:
        raise ValueError("date_range.end 不能早于 date_range.start")

    raw_re = re.compile(naming["raw_filename_regex"], re.IGNORECASE)
    merged_re = re.compile(naming["merged_filename_regex"], re.IGNORECASE)
    recursive = bool(naming.get("recursive", True))
    extensions = {ext.lower() for ext in naming.get("extensions", [".rnx"])}
    duplicate_policy = naming.get("duplicate_policy", "error").lower()
    records: Dict[Tuple[str, Optional[int], int, int], FileRecord] = {}

    def scan(folder: Path, dataset: str) -> None:
        if not folder.exists():
            LOGGER.warning("输入目录不存在：%s", folder)
            return
        iterator = folder.rglob("*") if recursive else folder.glob("*")
        for path in iterator:
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            if dataset == "merged" and raw_re.fullmatch(path.name):
                continue
            match = raw_re.fullmatch(path.name) if dataset == "raw" else merged_re.search(path.name)
            if not match:
                continue
            groups = match.groupdict()
            year, doy = int(groups["year"]), int(groups["doy"])
            try:
                date = datetime(year, 1, 1) + timedelta(days=doy - 1)
                if date.year != year:
                    continue
            except ValueError:
                continue
            if date < start or date > end:
                continue
            antenna = int(groups["antenna"]) if dataset == "raw" else None
            dataset_name = f"ant{antenna}" if dataset == "raw" else "merged"
            record = FileRecord(dataset_name, antenna, year, doy, date, path)
            key = (dataset_name, antenna, year, doy)
            if key in records:
                records[key] = choose_duplicate(records[key], record, duplicate_policy)
            else:
                records[key] = record

    scan(paths["raw_dir"], "raw")
    scan(paths["merged_dir"], "merged")
    return records


def choose_duplicate(old: FileRecord, new: FileRecord, policy: str) -> FileRecord:
    if policy == "latest":
        return max((old, new), key=lambda item: item.path.stat().st_mtime)
    if policy == "largest":
        return max((old, new), key=lambda item: item.path.stat().st_size)
    raise RuntimeError(f"同一数据集同一天匹配到多个文件：\n  {old.path}\n  {new.path}\n可在配置中设置 duplicate_policy=latest/largest。")


def print_discovery(records: Mapping[Tuple[str, Optional[int], int, int], FileRecord], config: Mapping[str, Any]) -> None:
    LOGGER.info("分析日期：%s 至 %s", config["date_range"]["start"], config["date_range"]["end"])
    days = sorted({(record.year, record.doy) for record in records.values()})
    for year, doy in days:
        found = []
        for name, antenna in (("ant1", 1), ("ant2", 2), ("merged", None)):
            rec = records.get((name, antenna, year, doy))
            found.append(f"{name}={'OK' if rec else '--'}")
        LOGGER.info("%04d-%03d  %s", year, doy, "  ".join(found))
    LOGGER.info("共发现 %d 个日期、%d 个文件。", len(days), len(records))


def read_rinex3_header_and_position(path: Path) -> Tuple[Dict[str, Any], int]:
    header = readObsHead(str(path), needSatList=False)
    if not header:
        raise RinexFormatError(f"无法读取 RINEX 头：{path}")
    version = float(header.get("version") or 0.0)
    if not 3.0 <= version < 4.0:
        raise RinexFormatError(f"当前分析程序面向 RINEX 3.x；{path.name} 的版本为 {version}")
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            if "END OF HEADER" in line:
                return header, line_no
    raise RinexFormatError(f"未找到 END OF HEADER：{path}")


def parse_obs_field(field: str) -> ObsValue:
    field = field.ljust(16)
    value_text = field[:14].strip()
    value: Optional[float] = None
    if value_text:
        try:
            parsed = float(value_text)
            if parsed != 0.0 and parsed > -999999999.0:
                value = parsed
        except ValueError:
            value = None
    lli = int(field[14]) if field[14:15].isdigit() else 0
    ssi = int(field[15]) if field[15:16].isdigit() else 0
    return ObsValue(value, lli, ssi)


def normalize_sat_id(raw: str) -> str:
    sat = raw.strip()
    if len(sat) == 2 and sat[0].isalpha():
        sat = sat[0] + sat[1:].zfill(2)
    elif len(sat) == 3 and sat[1] == " ":
        sat = sat[0] + "0" + sat[2]
    return sat


def iter_rinex3_epochs(path: Path, header: Mapping[str, Any], header_lines: int) -> Iterator[Tuple[datetime, int, Dict[str, Dict[str, ObsValue]]]]:
    obs_types_by_sys: Mapping[str, Sequence[str]] = header["OBS TYPES"]
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for _ in range(header_lines):
            next(f, None)
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith(">"):
                continue
            epoch = readRinex3LineTime(line)
            if epoch is None:
                continue
            tokens = line[1:].split()
            if len(tokens) < 8:
                continue
            try:
                flag, sat_num = int(tokens[6]), int(tokens[7])
            except ValueError:
                continue
            if flag not in (0, 1):
                for _ in range(sat_num):
                    next(f, None)
                continue

            epoch_obs: Dict[str, Dict[str, ObsValue]] = {}
            for _ in range(sat_num):
                first = f.readline()
                if not first:
                    break
                sat = normalize_sat_id(first[:3])
                if not sat:
                    continue
                system = sat[0]
                obs_types = list(obs_types_by_sys.get(system, []))
                if not obs_types:
                    continue
                payload = first[3:].rstrip("\r\n")
                needed = 16 * len(obs_types)
                standard_lines = max(1, math.ceil(len(obs_types) / 4))
                if len(payload) < needed:
                    for _continuation in range(standard_lines - 1):
                        continuation = f.readline()
                        if not continuation:
                            break
                        payload += continuation[3:].rstrip("\r\n")
                payload = payload.ljust(needed)
                values = {obs_type: parse_obs_field(payload[i * 16:(i + 1) * 16]) for i, obs_type in enumerate(obs_types)}
                epoch_obs[sat] = values
            yield epoch, flag, epoch_obs


def estimate_interval(epochs: Sequence[datetime], fallback: float) -> float:
    if len(epochs) < 2:
        return fallback
    diffs = [(b - a).total_seconds() for a, b in zip(epochs[:-1], epochs[1:])]
    diffs = [v for v in diffs if 0 < v <= 300]
    return float(median(diffs)) if diffs else fallback


def valid_code_phase_pairs(obs: Mapping[str, ObsValue]) -> List[str]:
    pairs = []
    for code_type, code in obs.items():
        if not code_type.startswith("C") or code.value is None:
            continue
        phase_type = "L" + code_type[1:]
        if phase_type in obs and obs[phase_type].value is not None:
            pairs.append(code_type[1:])
    return pairs


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def analyze_file(record: FileRecord, config: Mapping[str, Any]) -> DatasetResult:
    analysis = config["analysis"]
    systems: Set[str] = set(analysis.get("systems", ["G", "C"]))
    validity_mode = analysis.get("satellite_validity", "phase").lower()
    nominal_interval = float(analysis.get("nominal_interval_seconds", 10.0))
    gap_seconds_cfg = analysis.get("arc_gap_seconds")
    gap_factor = float(analysis.get("arc_gap_factor", 1.5))
    split_on_lli = bool(analysis.get("split_arc_on_lli", True))
    split_on_epoch_flag = bool(analysis.get("split_arc_on_epoch_flag", True))
    short_arc_seconds = float(analysis.get("short_arc_seconds", 300.0))
    minimum_satellites = int(analysis.get("minimum_satellites", 6))
    snr_min, snr_max = map(float, analysis.get("snr_range", [10.0, 70.0]))

    LOGGER.info("读取 %s：%s", record.dataset, record.path)
    header, header_lines = read_rinex3_header_and_position(record.path)
    header_interval = float(header.get("interval") or nominal_interval)
    gap_threshold = float(gap_seconds_cfg) if gap_seconds_cfg is not None else header_interval * gap_factor

    epoch_valid_sats: Dict[datetime, Set[str]] = {}
    epoch_recorded_sats: Dict[datetime, Set[str]] = {}
    snr_values: Dict[str, List[float]] = defaultdict(list)
    phase_obs_count = pair_obs_count = lli_event_count = 0
    active_arcs: Dict[Tuple[str, str], Dict[str, Any]] = {}
    arcs: List[Dict[str, Any]] = []

    def close_arc(key: Tuple[str, str]) -> None:
        arc = active_arcs.pop(key, None)
        if not arc:
            return
        arc["span_seconds"] = (arc["end_time"] - arc["start_time"]).total_seconds()
        arc["duration_seconds"] = arc["span_seconds"] + header_interval
        arcs.append(arc)

    for epoch, epoch_flag, epoch_obs in iter_rinex3_epochs(record.path, header, header_lines):
        if split_on_epoch_flag and epoch_flag == 1:
            for key in list(active_arcs):
                close_arc(key)
        recorded: Set[str] = set()
        valid: Set[str] = set()
        current_phase_keys: Set[Tuple[str, str]] = set()

        for sat, obs in epoch_obs.items():
            system = sat[0]
            if system not in systems:
                continue
            recorded.add(sat)
            phase_types = [name for name, item in obs.items() if name.startswith("L") and item.value is not None]
            pair_suffixes = valid_code_phase_pairs(obs)
            if validity_mode == "phase":
                is_valid = bool(phase_types)
            elif validity_mode == "code_phase_pair":
                is_valid = bool(pair_suffixes)
            elif validity_mode == "any":
                is_valid = any(item.value is not None for item in obs.values())
            else:
                raise ValueError(f"不支持的 satellite_validity：{validity_mode}")
            if is_valid:
                valid.add(sat)

            phase_obs_count += len(phase_types)
            pair_obs_count += len(pair_suffixes)
            for obs_type, item in obs.items():
                if obs_type.startswith("S") and item.value is not None and snr_min <= item.value <= snr_max:
                    snr_values[system].append(item.value)

            for phase_type in phase_types:
                item = obs[phase_type]
                key = (sat, phase_type)
                current_phase_keys.add(key)
                existing = active_arcs.get(key)
                split = existing is not None and (
                    (epoch - existing["end_time"]).total_seconds() > gap_threshold
                    or (split_on_lli and item.lli != 0)
                )
                if split:
                    close_arc(key)
                    existing = None
                if existing is None:
                    active_arcs[key] = {
                        "date": record.day_id,
                        "dataset": record.dataset,
                        "satellite": sat,
                        "system": system,
                        "phase_type": phase_type,
                        "start_time": epoch,
                        "end_time": epoch,
                        "sample_count": 1,
                        "start_lli": item.lli,
                    }
                else:
                    existing["end_time"] = epoch
                    existing["sample_count"] += 1
                if item.lli != 0:
                    lli_event_count += 1

        epoch_recorded_sats[epoch] = recorded
        epoch_valid_sats[epoch] = valid

        # 对当前历元未延续、且已经超过间隔阈值的弧段及时关闭，避免跨长空白错误连接。
        for key, arc in list(active_arcs.items()):
            if key not in current_phase_keys and (epoch - arc["end_time"]).total_seconds() > gap_threshold:
                close_arc(key)

    for key in list(active_arcs):
        close_arc(key)

    epochs = sorted(epoch_valid_sats)
    interval_s = estimate_interval(epochs, header_interval)
    valid_counts = [len(epoch_valid_sats[t]) for t in epochs]
    recorded_counts = [len(epoch_recorded_sats[t]) for t in epochs]
    start_time = epochs[0] if epochs else None
    end_time = epochs[-1] if epochs else None
    expected_epochs = round((end_time - start_time).total_seconds() / interval_s) + 1 if start_time and end_time and interval_s > 0 else len(epochs)
    durations = [arc["duration_seconds"] for arc in arcs]

    summary: Dict[str, Any] = {
        "date": record.day_id,
        "year": record.year,
        "doy": record.doy,
        "dataset": record.dataset,
        "file": str(record.path),
        "rinex_version": header.get("version"),
        "marker_name": header.get("MARKER NAME"),
        "start_time": start_time.isoformat(sep=" ") if start_time else "",
        "end_time": end_time.isoformat(sep=" ") if end_time else "",
        "interval_s": interval_s,
        "epoch_count": len(epochs),
        "expected_epoch_count_in_observed_span": expected_epochs,
        "epoch_coverage_rate": safe_ratio(len(epochs), expected_epochs),
        "mean_recorded_sat": safe_mean(recorded_counts),
        "mean_valid_sat": safe_mean(valid_counts),
        "min_valid_sat": min(valid_counts) if valid_counts else float("nan"),
        "p05_valid_sat": percentile(valid_counts, 5),
        "p95_valid_sat": percentile(valid_counts, 95),
        "max_valid_sat": max(valid_counts) if valid_counts else float("nan"),
        "low_sat_epoch_rate": safe_ratio(sum(v < minimum_satellites for v in valid_counts), len(valid_counts)),
        "phase_observation_count": phase_obs_count,
        "code_phase_pair_count": pair_obs_count,
        "lli_event_count": lli_event_count,
        "arc_count": len(arcs),
        "mean_arc_duration_s": safe_mean(durations),
        "median_arc_duration_s": percentile(durations, 50),
        "p95_arc_duration_s": percentile(durations, 95),
        "short_arc_rate": safe_ratio(sum(v < short_arc_seconds for v in durations), len(durations)),
    }
    for system in sorted(systems):
        system_counts = [sum(s.startswith(system) for s in epoch_valid_sats[t]) for t in epochs]
        system_snr = snr_values.get(system, [])
        summary[f"mean_valid_sat_{system}"] = safe_mean(system_counts)
        summary[f"p05_valid_sat_{system}"] = percentile(system_counts, 5)
        summary[f"snr_mean_{system}"] = safe_mean(system_snr)
        summary[f"snr_std_{system}"] = float(np.std(system_snr)) if system_snr else float("nan")
        summary[f"snr_count_{system}"] = len(system_snr)

    return DatasetResult(record, header, interval_s, epoch_valid_sats, epoch_recorded_sats, arcs, dict(snr_values), summary)


def make_epoch_comparison(day_id: str, results: Mapping[str, DatasetResult], systems: Sequence[str], minimum_satellites: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    all_times = sorted(set().union(*(set(result.epoch_valid_sats) for result in results.values())))
    rows: List[Dict[str, Any]] = []
    totals: Dict[str, float] = defaultdict(float)
    low_epochs: Dict[str, int] = defaultdict(int)

    for epoch in all_times:
        sets = {name: result.epoch_valid_sats.get(epoch, set()) for name, result in results.items()}
        ant1, ant2, merged = sets.get("ant1", set()), sets.get("ant2", set()), sets.get("merged", set())
        union, intersection = ant1 | ant2, ant1 & ant2
        row: Dict[str, Any] = {
            "date": day_id,
            "time": epoch.isoformat(sep=" "),
            "ant1": len(ant1),
            "ant2": len(ant2),
            "raw_union": len(union),
            "raw_intersection": len(intersection),
            "ant1_only": len(ant1 - ant2),
            "ant2_only": len(ant2 - ant1),
            "merged": len(merged),
            "merged_in_raw_union": len(merged & union),
            "merged_missing_from_union": len(union - merged),
            "merged_extra": len(merged - union),
        }
        for system in systems:
            for key, sat_set in (("ant1", ant1), ("ant2", ant2), ("raw_union", union), ("merged", merged)):
                row[f"{key}_{system}"] = sum(s.startswith(system) for s in sat_set)
        rows.append(row)
        for key in ("ant1", "ant2", "raw_union", "raw_intersection", "ant1_only", "ant2_only", "merged", "merged_in_raw_union", "merged_missing_from_union", "merged_extra"):
            totals[key] += row[key]
        for key in ("ant1", "ant2", "raw_union", "merged"):
            if row[key] < minimum_satellites:
                low_epochs[key] += 1

    n = len(rows)
    summary: Dict[str, Any] = {
        "date": day_id,
        "epoch_union_count": n,
        "mean_sat_ant1": safe_ratio(totals["ant1"], n),
        "mean_sat_ant2": safe_ratio(totals["ant2"], n),
        "mean_sat_raw_union": safe_ratio(totals["raw_union"], n),
        "mean_sat_merged": safe_ratio(totals["merged"], n),
        "raw_overlap_sat_epoch_rate": safe_ratio(totals["raw_intersection"], totals["raw_union"]),
        "raw_ant1_only_sat_epoch_rate": safe_ratio(totals["ant1_only"], totals["raw_union"]),
        "raw_ant2_only_sat_epoch_rate": safe_ratio(totals["ant2_only"], totals["raw_union"]),
        "raw_union_gain_vs_best_single": safe_ratio(totals["raw_union"], max(totals["ant1"], totals["ant2"])) - 1 if max(totals["ant1"], totals["ant2"]) else float("nan"),
        "merged_recall_of_raw_union": safe_ratio(totals["merged_in_raw_union"], totals["raw_union"]),
        "merged_extra_rate": safe_ratio(totals["merged_extra"], totals["merged"]),
        "merged_gain_vs_ant1": safe_ratio(totals["merged"], totals["ant1"]) - 1 if totals["ant1"] else float("nan"),
        "merged_gain_vs_ant2": safe_ratio(totals["merged"], totals["ant2"]) - 1 if totals["ant2"] else float("nan"),
        "low_sat_rate_ant1": safe_ratio(low_epochs["ant1"], n),
        "low_sat_rate_ant2": safe_ratio(low_epochs["ant2"], n),
        "low_sat_rate_raw_union": safe_ratio(low_epochs["raw_union"], n),
        "low_sat_rate_merged": safe_ratio(low_epochs["merged"], n),
    }
    for system in systems:
        for key in ("ant1", "ant2", "raw_union", "merged"):
            summary[f"mean_sat_{key}_{system}"] = safe_mean([row[f"{key}_{system}"] for row in rows])
    return rows, summary


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen: Set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_csv_value(row.get(key)) for key in fieldnames})


def serialize_csv_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def plot_satellite_counts(day_id: str, results: Mapping[str, DatasetResult], path: Path) -> None:
    times = sorted(set().union(*(set(result.epoch_valid_sats) for result in results.values())))
    if not times:
        return
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for name in ("ant1", "ant2", "merged"):
        result = results.get(name)
        if result:
            ax.plot(times, [len(result.epoch_valid_sats.get(t, set())) for t in times], label=name, linewidth=1.0)
    if "ant1" in results and "ant2" in results:
        ax.plot(times, [len(results["ant1"].epoch_valid_sats.get(t, set()) | results["ant2"].epoch_valid_sats.get(t, set())) for t in times], label="raw_union", linewidth=1.0)
    ax.set_title(f"Visible satellites (valid carrier phase) - {day_id}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Satellite count")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_arc_ecdf(day_id: str, results: Mapping[str, DatasetResult], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    plotted = False
    for name in ("ant1", "ant2", "merged"):
        result = results.get(name)
        durations = sorted(arc["duration_seconds"] / 60.0 for arc in result.arcs) if result else []
        if not durations:
            continue
        y = np.arange(1, len(durations) + 1) / len(durations)
        ax.plot(durations, y, label=name, linewidth=1.2)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_title(f"Carrier-phase arc duration ECDF - {day_id}")
    ax.set_xlabel("Arc duration (min)")
    ax.set_ylabel("Cumulative probability")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def downsample(values: Sequence[float], maximum: int) -> List[float]:
    if len(values) <= maximum:
        return list(values)
    step = math.ceil(len(values) / maximum)
    return list(values[::step])


def plot_snr_boxplot(day_id: str, results: Mapping[str, DatasetResult], systems: Sequence[str], path: Path, maximum: int) -> None:
    data, labels = [], []
    for name in ("ant1", "ant2", "merged"):
        result = results.get(name)
        if not result:
            continue
        for system in systems:
            values = result.snr_values.get(system, [])
            if values:
                data.append(downsample(values, maximum))
                labels.append(f"{name}-{system}")
    if not data:
        return
    fig, ax = plt.subplots(figsize=(max(8.5, len(labels) * 1.2), 5.5))
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_title(f"SNR distribution - {day_id}")
    ax.set_xlabel("Dataset-system")
    ax.set_ylabel("SNR (dB-Hz)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_daily_overview(dataset_summaries: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    if not dataset_summaries:
        return
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in dataset_summaries:
        grouped[str(row["dataset"])].append(row)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name in ("ant1", "ant2", "merged"):
        rows = sorted(grouped.get(name, []), key=lambda item: str(item["date"]))
        if rows:
            dates = [datetime.strptime(str(row["date"]), "%Y%j") for row in rows]
            ax.plot(dates, [float(row["mean_valid_sat"]) for row in rows], marker="o", label=name)
    ax.set_title("Daily mean visible satellites")
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean satellite count")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "daily_mean_satellites.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name in ("ant1", "ant2", "merged"):
        rows = sorted(grouped.get(name, []), key=lambda item: str(item["date"]))
        if rows:
            dates = [datetime.strptime(str(row["date"]), "%Y%j") for row in rows]
            ax.plot(dates, [float(row["median_arc_duration_s"]) / 60.0 for row in rows], marker="o", label=name)
    ax.set_title("Daily median carrier-phase arc duration")
    ax.set_xlabel("Date")
    ax.set_ylabel("Median arc duration (min)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "daily_median_arc_duration.png", dpi=220)
    plt.close(fig)


def run(config: Mapping[str, Any], dry_run: bool = False) -> None:
    records = discover_files(config)
    print_discovery(records, config)
    if dry_run:
        return
    if not records:
        raise RuntimeError("在指定日期和目录中没有发现匹配的 RINEX 文件。")

    output_dir: Path = config["paths"]["output_dir"]
    (output_dir / "summary").mkdir(parents=True, exist_ok=True)
    (output_dir / "epochs").mkdir(parents=True, exist_ok=True)
    (output_dir / "arcs").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    analysis_cfg = config["analysis"]
    systems = list(analysis_cfg.get("systems", ["G", "C"]))
    minimum_satellites = int(analysis_cfg.get("minimum_satellites", 6))
    plot_cfg = config.get("plots", {})
    dataset_summaries: List[Dict[str, Any]] = []
    comparison_summaries: List[Dict[str, Any]] = []

    days = sorted({(record.year, record.doy) for record in records.values()})
    for year, doy in days:
        day_id = f"{year}{doy:03d}"
        day_results: Dict[str, DatasetResult] = {}
        for name, antenna in (("ant1", 1), ("ant2", 2), ("merged", None)):
            record = records.get((name, antenna, year, doy))
            if not record:
                continue
            result = analyze_file(record, config)
            day_results[name] = result
            dataset_summaries.append(result.summary)

        all_arcs = [arc for result in day_results.values() for arc in result.arcs]
        write_csv(output_dir / "arcs" / f"{day_id}_arc_metrics.csv", all_arcs)

        if len(day_results) >= 2:
            epoch_rows, comparison = make_epoch_comparison(day_id, day_results, systems, minimum_satellites)
            comparison_summaries.append(comparison)
            if analysis_cfg.get("write_epoch_csv", True):
                write_csv(output_dir / "epochs" / f"{day_id}_epoch_metrics.csv", epoch_rows)

        if plot_cfg.get("satellite_count", True):
            plot_satellite_counts(day_id, day_results, output_dir / "figures" / f"{day_id}_satellite_count.png")
        if plot_cfg.get("arc_ecdf", True):
            plot_arc_ecdf(day_id, day_results, output_dir / "figures" / f"{day_id}_arc_duration_ecdf.png")
        if plot_cfg.get("snr_boxplot", True):
            plot_snr_boxplot(day_id, day_results, systems, output_dir / "figures" / f"{day_id}_snr_boxplot.png", int(plot_cfg.get("max_snr_points_per_group", 20000)))

    write_csv(output_dir / "summary" / "daily_dataset_summary.csv", dataset_summaries)
    write_csv(output_dir / "summary" / "daily_comparison_summary.csv", comparison_summaries)
    plot_daily_overview(dataset_summaries, output_dir)
    LOGGER.info("分析完成，结果目录：%s", output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="分析双天线原始 RINEX 与融合 RINEX 的观测质量")
    parser.add_argument("--config", required=True, type=Path, help="JSON 配置文件路径")
    parser.add_argument("--dry-run", action="store_true", help="仅检查文件匹配情况，不读取 RINEX")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    config = load_config(args.config.resolve())
    run(config, args.dry_run)


if __name__ == "__main__":
    main()
