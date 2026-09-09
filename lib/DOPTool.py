"""Reusable geometry-DOP utilities for GNSS and LEO POD analyses."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import re

import numpy as np
import pandas as pd


def iter_rinex_observed_prns(path: Path):
    """Yield RINEX-3 epochs and PRNs having at least one non-empty observation."""
    in_header, current, satellites = True, None, set()
    with path.open("r", encoding="ascii", errors="ignore") as handle:
        for line in handle:
            if in_header:
                in_header = "END OF HEADER" not in line
                continue
            if line.startswith(">"):
                if current is not None:
                    yield current, satellites
                parts = line[1:].split(); second = float(parts[5])
                current = datetime(*map(int, parts[:5]), int(second), round((second % 1) * 1_000_000))
                satellites = set()
            elif current is not None and len(line) >= 3 and line[0] in "GCEJR":
                if any(line[index:index + 14].strip() for index in range(3, len(line), 16)):
                    satellites.add(line[:3].strip())
    if current is not None:
        yield current, satellites


def rinex_day_tag(path: Path) -> str:
    """Return ``YYYYMMDD`` from a RINEX filename containing ``YYYYDDD``."""
    match = re.search(r"(2024\d{3}|2025\d{3})", path.name)
    return datetime.strptime(match.group(1), "%Y%j").strftime("%Y%m%d") if match else path.stem


def read_sp3_positions(paths: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read one or more SP3 files as ``{PRN: (unix_seconds, ECEF_m)}``."""
    records = defaultdict(list)
    for raw_path in paths:
        epoch = None
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"SP3 product not found: {path}")
        for line in path.open("r", encoding="ascii", errors="ignore"):
            if line.startswith("*"):
                p = line[1:].split(); second = float(p[5])
                epoch = datetime(*map(int, p[:5]), int(second), round((second % 1) * 1_000_000))
            elif epoch is not None and line.startswith("P") and len(line) >= 46:
                try:
                    sat = line[1:4].strip()
                    xyz = [float(line[4:18]) * 1000, float(line[18:32]) * 1000, float(line[32:46]) * 1000]
                except ValueError:
                    continue
                if sat and sat != "0" and all(abs(value) < 9e11 for value in xyz):
                    records[sat].append((epoch.timestamp(), xyz))
    product = {}
    for sat, values in records.items():
        values.sort(key=lambda item: item[0])
        times, xyz = zip(*values)
        unique = pd.DataFrame(xyz, index=np.asarray(times)).groupby(level=0).last()
        product[sat] = unique.index.to_numpy(float), unique.to_numpy(float)
    return product


def interpolate_position(product, satellite: str, timestamp: float) -> np.ndarray | None:
    """Linearly interpolate an ECEF position; return None outside product coverage."""
    data = product.get(satellite)
    if data is None:
        return None
    times, xyz = data
    if timestamp < times[0] or timestamp > times[-1]:
        return None
    return np.array([np.interp(timestamp, times, xyz[:, component]) for component in range(3)])


def calculate_rtn_matrix(receiver: np.ndarray, reference_product, receiver_id: str, timestamp: float,
                         velocity_step_s: float = 15.0) -> np.ndarray | None:
    """Return ECEF-to-RTN rotation matrix based on a reference LEO orbit."""
    before = interpolate_position(reference_product, receiver_id, timestamp - velocity_step_s)
    after = interpolate_position(reference_product, receiver_id, timestamp + velocity_step_s)
    if before is None or after is None:
        return None
    radial = receiver / np.linalg.norm(receiver)
    normal = np.cross(receiver, after - before)
    normal /= np.linalg.norm(normal)
    along = np.cross(normal, radial)
    return np.vstack((radial, along, normal))


def calculate_dop(receiver_ecef: np.ndarray, satellite_ecef: list[np.ndarray],
                  ecef_to_rtn: np.ndarray | None = None) -> dict[str, float] | None:
    """Calculate GDOP/PDOP/TDOP and optional RTN directional DOP.

    ``None`` is returned for fewer than four satellites or a rank-deficient
    geometry matrix. The returned DOP values are dimensionless.
    """
    if len(satellite_ecef) < 4:
        return None
    los = np.asarray([(sat - receiver_ecef) / np.linalg.norm(sat - receiver_ecef) for sat in satellite_ecef])
    design = np.column_stack((-los, np.ones(len(los))))
    normal = design.T @ design
    if np.linalg.matrix_rank(normal) < 4:
        return None
    covariance = np.linalg.inv(normal)
    position_covariance = covariance[:3, :3]
    result = {"gdop": float(np.sqrt(np.trace(covariance))), "pdop": float(np.sqrt(np.trace(position_covariance))),
              "tdop": float(np.sqrt(covariance[3, 3]))}
    if ecef_to_rtn is None:
        result.update(rdop=np.nan, adop=np.nan, cdop=np.nan)
    else:
        rtn_covariance = ecef_to_rtn @ position_covariance @ ecef_to_rtn.T
        result.update(rdop=float(np.sqrt(rtn_covariance[0, 0])), adop=float(np.sqrt(rtn_covariance[1, 1])),
                      cdop=float(np.sqrt(rtn_covariance[2, 2])))
    return result
