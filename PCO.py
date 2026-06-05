"""
APC to CoM correction for dual-antenna GNSS observations.

This module implements the simplified correction model:

    d_B = CoM -> APC offset in spacecraft body frame, configured in metres
    d_O = C_OB @ d_B
    d_I = C_IO @ d_O

where O is the RTN/orbit frame built from the LEO CoM position and velocity,
B is the spacecraft body frame, and I is the inertial frame used by the SP3
files. The preferred attitude mode for CSU attitude files is inertial_to_body,
matching the C++ CAttitude::ToCosineMatrix convention. The corrected
observations are normalized from APC to CoM with the first-order range correction

    delta = dot(unit(GNSS - LEO), d_I)

Code observations receive delta in metres; carrier phase observations receive
delta / wavelength in cycles.
"""

from __future__ import annotations

import copy
import datetime as _dt
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

import AttitudeModel
import Rinex
import Sp3Tools


C_MPS = 299792458.0

GNSS_OBS_FREQUENCIES_HZ: Dict[str, Dict[str, float]] = {
    "G": {
        "1": 1575.42e6,
        "2": 1227.60e6,
        "5": 1176.45e6,
    },
    "C": {
        # BeiDou RINEX bands used in GNSS_MIX:
        # L2 -> B1, L6 -> B3, L1 -> B1C/B1A, L5 -> B2a, L7 -> B2/B2b, L8 -> B2a+B2b.
        "1": 1575.42e6,
        "2": 1561.098e6,
        "5": 1176.45e6,
        "6": 1268.52e6,
        "7": 1207.140e6,
        "8": 1191.795e6,
    },
    "E": {
        "1": 1575.42e6,
        "5": 1176.45e6,
        "6": 1278.75e6,
        "7": 1207.140e6,
        "8": 1191.795e6,
    },
    "S": {
        "1": 1575.42e6,
        "5": 1176.45e6,
    },
    "J": {
        "1": 1575.42e6,
        "2": 1227.60e6,
        "5": 1176.45e6,
        "6": 1278.75e6,
    },
    "I": {
        "5": 1176.45e6,
        "9": 2492.028e6,
    },
}


@dataclass(frozen=True)
class CorrectionContext:
    leo_index: Mapping[str, Mapping[str, Any]]
    gnss_index: Mapping[Tuple[str, str], Mapping[str, Any]]
    attitude_index: Mapping[str, AttitudeModel.Attitude]
    leo_series: Mapping[str, np.ndarray]
    gnss_series: Mapping[str, Mapping[str, np.ndarray]]
    leo_times: np.ndarray
    gnss_times: np.ndarray
    attitude_times: np.ndarray
    attitude_values: Tuple[AttitudeModel.Attitude, ...]
    atx_pco_index: Mapping[str, Mapping[str, np.ndarray]]
    attitude_reference_frame: str
    attitude_matrix_direction: str
    gnss_pco_frame: str
    time_key_decimals: int
    sp3_interpolation_max_gap_s: float
    sp3_epoch_match_tolerance_s: float
    attitude_nearest_max_gap_s: float


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    required = ["leo_sp3_file", "gnss_sp3_file", "attitude_file", "antennas"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    for sp3_key in ["leo_sp3_file", "gnss_sp3_file"]:
        if isinstance(config[sp3_key], list) and not config[sp3_key]:
            raise ValueError(f"Config key '{sp3_key}' must not be an empty list.")
    if float(config.get("sp3_epoch_match_tolerance_s", 10.0)) < 0.0:
        raise ValueError("sp3_epoch_match_tolerance_s must be non-negative.")
    reference_frame = config.get("attitude_reference_frame", "inertial").lower()
    if reference_frame not in {"orbit", "inertial"}:
        raise ValueError(
            "attitude_reference_frame must be 'orbit' or 'inertial'."
        )
    direction = config.get("attitude_matrix_direction", "inertial_to_body").lower()
    allowed_directions = {
        "body_to_orbit",
        "orbit_to_body",
        "body_to_inertial",
        "inertial_to_body",
    }
    if direction not in allowed_directions:
        raise ValueError(
            "attitude_matrix_direction must be one of "
            "'body_to_orbit', 'orbit_to_body', 'body_to_inertial', "
            "or 'inertial_to_body'."
        )
    if reference_frame == "orbit" and direction not in {"body_to_orbit", "orbit_to_body"}:
        raise ValueError("Orbit attitude requires body_to_orbit or orbit_to_body direction.")
    if reference_frame == "inertial" and direction not in {"body_to_inertial", "inertial_to_body"}:
        raise ValueError("Inertial attitude requires body_to_inertial or inertial_to_body direction.")
    pco_frame = config.get("gnss_pco_frame", "rtn_nadir").lower()
    if pco_frame not in {"rtn_nadir", "none"}:
        raise ValueError("gnss_pco_frame must be 'rtn_nadir' or 'none'.")
    if not config["antennas"]:
        raise ValueError("Config key 'antennas' must contain at least one antenna.")
    for antenna in config["antennas"]:
        for key in ["name", "input_rinex", "output_rinex"]:
            if key not in antenna:
                raise ValueError(f"Antenna config missing '{key}': {antenna}")
        has_body = "com_to_apc_body_m" in antenna
        has_orbit = "com_to_apc_orbit_m" in antenna
        if not has_body and not has_orbit:
            raise ValueError(
                "Antenna config must contain either 'com_to_apc_body_m' "
                f"or 'com_to_apc_orbit_m': {antenna}"
            )
        if has_body and len(antenna["com_to_apc_body_m"]) != 3:
            raise ValueError(f"com_to_apc_body_m must have 3 elements: {antenna}")
        if has_orbit and len(antenna["com_to_apc_orbit_m"]) != 3:
            raise ValueError(f"com_to_apc_orbit_m must have 3 elements: {antenna}")


def time_key(value: Any, decimals: int = 3) -> str:
    """Return a stable key for TimeSystem or datetime epochs."""
    dt = getattr(value, "datetime", value)
    if not isinstance(dt, _dt.datetime):
        raise TypeError(f"Unsupported time object: {type(value)}")
    scale = 10**decimals
    whole = int(round(dt.timestamp() * scale))
    return str(whole)


def epoch_seconds(value: Any) -> float:
    dt = getattr(value, "datetime", value)
    if not isinstance(dt, _dt.datetime):
        raise TypeError(f"Unsupported time object: {type(value)}")
    return float(dt.timestamp())


def attitude_time_offset_seconds(config: Mapping[str, Any]) -> float:
    if "attitude_time_offset_s" in config:
        return float(config["attitude_time_offset_s"])
    attitude_scale = config.get("attitude_time_scale", "GPS").upper()
    observation_scale = config.get("observation_time_scale", "GPS").upper()
    if attitude_scale == "UTC" and observation_scale == "GPS":
        return float(config.get("gps_utc_leap_seconds", 18.0))
    if attitude_scale == "GPS" and observation_scale == "UTC":
        return -float(config.get("gps_utc_leap_seconds", 18.0))
    return 0.0


def vector_norm(vec: np.ndarray, name: str) -> float:
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError(f"{name} has zero or invalid norm: {vec}")
    return norm


def orbit_frame_matrices(pos_i: Iterable[float], vel_i: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Build C_OI and C_IO from inertial position/velocity.

    C_OI maps inertial vectors to RTN/orbit vectors. Its rows are the RTN basis
    vectors expressed in the inertial frame. C_IO is the transpose and maps RTN
    vectors back to inertial coordinates.
    """
    r = np.asarray(pos_i, dtype=float)
    v = np.asarray(vel_i, dtype=float)

    u_r = r / vector_norm(r, "position")
    h = np.cross(r, v)
    u_n = h / vector_norm(h, "angular momentum")
    u_t = np.cross(u_n, u_r)
    u_t = u_t / vector_norm(u_t, "transverse axis")

    c_oi = np.vstack([u_r, u_t, u_n])
    c_io = c_oi.T
    return c_oi, c_io


def attitude_body_to_orbit(attitude: AttitudeModel.Attitude, direction: str) -> np.ndarray:
    matrix = attitude.get_rotation_matrix()
    direction = direction.lower()
    if direction == "body_to_orbit":
        return matrix
    if direction == "orbit_to_body":
        return matrix.T
    raise ValueError(
        "attitude_matrix_direction must be 'body_to_orbit' or 'orbit_to_body'."
    )


def attitude_body_to_inertial(attitude: AttitudeModel.Attitude, direction: str) -> np.ndarray:
    matrix = attitude.get_rotation_matrix()
    direction = direction.lower()
    if direction == "body_to_inertial":
        return matrix
    if direction == "inertial_to_body":
        return matrix.T
    raise ValueError(
        "attitude_matrix_direction must be 'body_to_inertial' or 'inertial_to_body'."
    )


def apc_offset_inertial(
    com_to_apc_body_m: Iterable[float],
    leo_pos_i: Iterable[float],
    leo_vel_i: Iterable[float],
    attitude: AttitudeModel.Attitude,
    attitude_matrix_direction: str = "body_to_orbit",
    attitude_reference_frame: str = "orbit",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return CoM -> APC offset in inertial frame, plus C_OI and C_OB."""
    d_b = np.asarray(com_to_apc_body_m, dtype=float)
    c_oi, c_io = orbit_frame_matrices(leo_pos_i, leo_vel_i)
    attitude_reference_frame = attitude_reference_frame.lower()
    if attitude_reference_frame == "orbit":
        c_ob = attitude_body_to_orbit(attitude, attitude_matrix_direction)
        d_o = c_ob @ d_b
        d_i = c_io @ d_o
        return d_i, c_oi, c_ob
    if attitude_reference_frame == "inertial":
        c_ib = attitude_body_to_inertial(attitude, attitude_matrix_direction)
        d_i = c_ib @ d_b
        return d_i, c_oi, c_ib
    raise ValueError("attitude_reference_frame must be 'orbit' or 'inertial'.")


def antenna_offset_inertial(
    antenna: Mapping[str, Any],
    leo_pos_i: Iterable[float],
    leo_vel_i: Iterable[float],
    attitude: AttitudeModel.Attitude,
    attitude_matrix_direction: str,
    attitude_reference_frame: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return configured antenna offset in inertial frame.

    The normal branch rotates a body-frame offset with the attitude model. Some
    external DRO ANT3 products instead use a fixed RTN/orbit-frame offset; that
    convention is selected by com_to_apc_orbit_m.
    """
    if "com_to_apc_orbit_m" in antenna:
        d_o = np.asarray(antenna["com_to_apc_orbit_m"], dtype=float)
        c_oi, c_io = orbit_frame_matrices(leo_pos_i, leo_vel_i)
        return c_io @ d_o, c_oi, np.eye(3)
    return apc_offset_inertial(
        antenna["com_to_apc_body_m"],
        leo_pos_i,
        leo_vel_i,
        attitude,
        attitude_matrix_direction,
        attitude_reference_frame,
    )


def line_of_sight_unit(leo_pos_i: Iterable[float], gnss_pos_i: Iterable[float]) -> np.ndarray:
    leo = np.asarray(leo_pos_i, dtype=float)
    gnss = np.asarray(gnss_pos_i, dtype=float)
    los = gnss - leo
    return los / vector_norm(los, "line of sight")


def range_correction_m(
    com_to_apc_body_m: Iterable[float],
    leo_pos_i: Iterable[float],
    leo_vel_i: Iterable[float],
    gnss_pos_i: Iterable[float],
    attitude: AttitudeModel.Attitude,
    attitude_matrix_direction: str = "body_to_orbit",
    attitude_reference_frame: str = "orbit",
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    d_i, c_oi, c_ob = apc_offset_inertial(
        com_to_apc_body_m,
        leo_pos_i,
        leo_vel_i,
        attitude,
        attitude_matrix_direction,
        attitude_reference_frame,
    )
    u = line_of_sight_unit(leo_pos_i, gnss_pos_i)
    delta = float(np.dot(u, d_i))
    return delta, d_i, c_oi, c_ob


def atx_frequency_code(prn: str, obs_type: str) -> Optional[str]:
    """Map a RINEX observation type to an ANTEX frequency code."""
    if len(prn) < 1 or len(obs_type) < 2:
        return None
    band = obs_type[1]
    if not band.isdigit():
        return None
    return f"{prn[0]}{int(band):02d}"


def gnss_pco_neu_m(
    atx_pco_index: Mapping[str, Mapping[str, np.ndarray]],
    prn: str,
    obs_type: str,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    sat_pcos = atx_pco_index.get(prn)
    if not sat_pcos:
        return None, None

    freq_code = atx_frequency_code(prn, obs_type)
    if freq_code and freq_code in sat_pcos:
        return sat_pcos[freq_code], freq_code

    if sat_pcos:
        fallback_freq = sorted(sat_pcos)[0]
        return sat_pcos[fallback_freq], fallback_freq
    return None, None


def gnss_pco_inertial_m(
    pco_neu_m: Optional[np.ndarray],
    gnss_pos_i: Iterable[float],
    gnss_vel_i: Iterable[float],
    frame: str = "rtn_nadir",
) -> np.ndarray:
    """Convert GNSS satellite ANTEX NEU PCO to inertial coordinates.

    In the default approximation, ANTEX Up is the Earth-pointing antenna axis.
    With RTN built from GNSS CoM position/velocity, this gives:
        North -> +N, East -> +T, Up -> -R
    """
    if pco_neu_m is None or frame == "none":
        return np.zeros(3)
    if frame != "rtn_nadir":
        raise ValueError(f"Unsupported gnss_pco_frame: {frame}")

    north, east, up = np.asarray(pco_neu_m, dtype=float)
    _, c_io = orbit_frame_matrices(gnss_pos_i, gnss_vel_i)
    pco_rtn_m = np.array([-up, east, north], dtype=float)
    return c_io @ pco_rtn_m


def as_list(value: Any) -> List[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def read_sp3_dataframe(paths: Any) -> pd.DataFrame:
    frames = []
    for path in as_list(paths):
        sp3 = Sp3Tools.SP3(path)
        df = sp3.asDataFrame()
        if df.empty:
            raise ValueError(f"No SP3 data found: {path}")
        df = df.copy()
        df["source_sp3"] = path
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["timesystem", "sat_id"], keep="last")
    return merged.sort_values(["sat_id", "gps_time"]).reset_index(drop=True)


def estimate_missing_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate vx/vy/vz by finite difference when SP3 has position only."""
    if {"vx", "vy", "vz"}.issubset(df.columns):
        return df

    frames = []
    for sat_id, group in df.sort_values("gps_time").groupby("sat_id", sort=False):
        g = group.copy()
        t = g["gps_time"].to_numpy(dtype=float)
        for axis in ["x", "y", "z"]:
            values = g[axis].to_numpy(dtype=float)
            if len(values) == 1:
                deriv = np.zeros_like(values)
            else:
                deriv = np.gradient(values, t)
            g["v" + axis] = deriv
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def select_leo_dataframe(df: pd.DataFrame, satellite: Optional[str]) -> pd.DataFrame:
    source_df = df
    original_satellite = satellite
    if satellite:
        df = df[df["sat_id"] == satellite].copy()
    if df.empty:
        unique_satellites = sorted(str(sat) for sat in source_df["sat_id"].unique())
        if original_satellite and len(unique_satellites) == 1:
            fallback_satellite = unique_satellites[0]
            print(
                "LEO satellite filter "
                f"{original_satellite!r} matched no rows; using the only SP3 satellite "
                f"{fallback_satellite!r}."
            )
            df = source_df[source_df["sat_id"] == fallback_satellite].copy()
        else:
            raise ValueError(
                "No LEO SP3 rows remain after applying satellite filter "
                f"{original_satellite!r}. Available satellite IDs: {unique_satellites}"
            )
    return df


def build_state_series(df: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
    df = estimate_missing_velocity(df)
    series: Dict[str, Dict[str, np.ndarray]] = {}
    for sat_id, group in df.sort_values("gps_time").groupby("sat_id", sort=False):
        g = group.copy()
        t = np.array([epoch_seconds(ts) for ts in g["timesystem"]], dtype=float)
        series[str(sat_id)] = {
            "time": t,
            "pos": g[["x", "y", "z"]].to_numpy(dtype=float),
            "vel": g[["vx", "vy", "vz"]].to_numpy(dtype=float),
        }
    return series


def build_leo_index(df: pd.DataFrame, satellite: Optional[str], decimals: int) -> Dict[str, Dict[str, Any]]:
    df = select_leo_dataframe(df, satellite)

    df = estimate_missing_velocity(df)
    index: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        key = time_key(row["timesystem"], decimals)
        index[key] = {
            "pos": np.array([row["x"], row["y"], row["z"]], dtype=float),
            "vel": np.array([row["vx"], row["vy"], row["vz"]], dtype=float),
            "sat_id": row["sat_id"],
        }
    return index


def build_leo_series(df: pd.DataFrame, satellite: Optional[str]) -> Dict[str, np.ndarray]:
    selected = select_leo_dataframe(df, satellite)
    state_series = build_state_series(selected)
    if len(state_series) != 1:
        raise ValueError("LEO series must contain exactly one satellite after filtering.")
    return next(iter(state_series.values()))


def build_gnss_index(df: pd.DataFrame, decimals: int) -> Dict[Tuple[str, str], Dict[str, Any]]:
    df = estimate_missing_velocity(df)
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for _, row in df.iterrows():
        key = time_key(row["timesystem"], decimals)
        sat_id = str(row["sat_id"]).strip()
        index[(key, sat_id)] = {
            "pos": np.array([row["x"], row["y"], row["z"]], dtype=float),
            "vel": np.array([row["vx"], row["vy"], row["vz"]], dtype=float),
            "sat_id": sat_id,
        }
    return index


def interpolate_state(
    series: Mapping[str, np.ndarray],
    target_time: float,
    max_gap_s: float,
) -> Optional[Dict[str, np.ndarray]]:
    times = series["time"]
    if len(times) == 0:
        return None
    if target_time < times[0]:
        if times[0] - target_time <= max_gap_s:
            return {"pos": series["pos"][0], "vel": series["vel"][0]}
        return None
    if target_time > times[-1]:
        if target_time - times[-1] <= max_gap_s:
            return {"pos": series["pos"][-1], "vel": series["vel"][-1]}
        return None

    right = int(np.searchsorted(times, target_time, side="left"))
    if right < len(times) and abs(times[right] - target_time) < 1e-6:
        return {"pos": series["pos"][right], "vel": series["vel"][right]}
    if right == 0 or right >= len(times):
        return None

    left = right - 1
    interval = times[right] - times[left]
    if interval <= 0.0 or interval > max_gap_s:
        return None
    ratio = (target_time - times[left]) / interval
    pos = series["pos"][left] * (1.0 - ratio) + series["pos"][right] * ratio
    vel = series["vel"][left] * (1.0 - ratio) + series["vel"][right] * ratio
    return {"pos": pos, "vel": vel}


def unique_epoch_seconds(df: pd.DataFrame) -> np.ndarray:
    times = np.array([epoch_seconds(ts) for ts in df["timesystem"]], dtype=float)
    return np.array(sorted(set(times)), dtype=float)


def nearest_time_difference_s(times: np.ndarray, target_time: float) -> float:
    if len(times) == 0:
        return float("inf")
    right = int(np.searchsorted(times, target_time, side="left"))
    diffs = []
    if right < len(times):
        diffs.append(abs(times[right] - target_time))
    if right > 0:
        diffs.append(abs(times[right - 1] - target_time))
    return min(diffs) if diffs else float("inf")


def within_sp3_epoch_tolerance(
    context: CorrectionContext,
    target_time: float,
) -> Tuple[bool, float, float]:
    leo_dt = nearest_time_difference_s(context.leo_times, target_time)
    gnss_dt = nearest_time_difference_s(context.gnss_times, target_time)
    tolerance = context.sp3_epoch_match_tolerance_s
    return leo_dt <= tolerance and gnss_dt <= tolerance, leo_dt, gnss_dt


def nearest_attitude(
    times: np.ndarray,
    attitudes: Tuple[AttitudeModel.Attitude, ...],
    target_time: float,
    max_gap_s: float,
) -> Optional[AttitudeModel.Attitude]:
    if len(times) == 0:
        return None
    right = int(np.searchsorted(times, target_time, side="left"))
    candidates = []
    if right < len(times):
        candidates.append(right)
    if right > 0:
        candidates.append(right - 1)
    best = min(candidates, key=lambda idx: abs(times[idx] - target_time))
    if abs(times[best] - target_time) > max_gap_s:
        return None
    return attitudes[best]


def parse_atx_pco(path: Optional[str]) -> Dict[str, Dict[str, np.ndarray]]:
    """Parse satellite PCO values from ANTEX.

    Returns {prn: {frequency_code: np.array([north, east, up]) metres}}.
    Receiver antenna blocks without a satellite PRN are ignored.
    """
    if not path:
        return {}

    pco_index: Dict[str, Dict[str, np.ndarray]] = {}
    current_prn: Optional[str] = None
    current_freq: Optional[str] = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            label = line[60:].strip() if len(line) >= 60 else ""
            body = line[:60]

            if label == "TYPE / SERIAL NO":
                fields = body.split()
                current_prn = None
                for field in fields:
                    if len(field) == 3 and field[0] in "GRECJIS" and field[1:].isdigit():
                        current_prn = field
                        pco_index.setdefault(current_prn, {})
                        break
            elif label == "START OF FREQUENCY":
                current_freq = body.split()[0] if body.split() else None
            elif label == "NORTH / EAST / UP" and current_prn and current_freq:
                parts = body.split()
                if len(parts) >= 3:
                    neu_m = np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float) / 1000.0
                    pco_index.setdefault(current_prn, {})[current_freq] = neu_m
            elif label == "END OF FREQUENCY":
                current_freq = None
            elif label == "END OF ANTENNA":
                current_prn = None
                current_freq = None

    return pco_index


def build_attitude_index(
    path: str,
    decimals: int,
    time_offset_s: float = 0.0,
) -> Dict[str, AttitudeModel.Attitude]:
    attitude_df = AttitudeModel.read_attitude_file(path)
    return {
        time_key(row["timesystem"] + time_offset_s, decimals): row["attitude"]
        for _, row in attitude_df.iterrows()
    }


def build_attitude_series(
    path: str,
    time_offset_s: float = 0.0,
) -> Tuple[np.ndarray, Tuple[AttitudeModel.Attitude, ...]]:
    attitude_df = AttitudeModel.read_attitude_file(path)
    attitude_df = attitude_df.sort_values("timesystem").reset_index(drop=True)
    times = np.array(
        [epoch_seconds(ts) + time_offset_s for ts in attitude_df["timesystem"]],
        dtype=float,
    )
    values = tuple(attitude_df["attitude"])
    return times, values


def prepare_context(config: Mapping[str, Any]) -> CorrectionContext:
    decimals = int(config.get("time_key_decimals", 3))
    att_time_offset_s = attitude_time_offset_seconds(config)
    leo_df = read_sp3_dataframe(config["leo_sp3_file"])
    gnss_df = read_sp3_dataframe(config["gnss_sp3_file"])
    leo_selected_df = select_leo_dataframe(leo_df, config.get("leo_satellite"))
    leo_series = build_leo_series(leo_selected_df, None)
    gnss_series = build_state_series(gnss_df)
    attitude_times, attitude_values = build_attitude_series(
        config["attitude_file"],
        att_time_offset_s,
    )
    return CorrectionContext(
        leo_index=build_leo_index(leo_selected_df, None, decimals),
        gnss_index=build_gnss_index(gnss_df, decimals),
        attitude_index=build_attitude_index(config["attitude_file"], decimals, att_time_offset_s),
        leo_series=leo_series,
        gnss_series=gnss_series,
        leo_times=unique_epoch_seconds(leo_selected_df),
        gnss_times=unique_epoch_seconds(gnss_df),
        attitude_times=attitude_times,
        attitude_values=attitude_values,
        atx_pco_index=parse_atx_pco(config.get("igs_atx")),
        attitude_reference_frame=config.get("attitude_reference_frame", "inertial"),
        attitude_matrix_direction=config.get("attitude_matrix_direction", "inertial_to_body"),
        gnss_pco_frame=config.get("gnss_pco_frame", "rtn_nadir"),
        time_key_decimals=decimals,
        sp3_interpolation_max_gap_s=float(config.get("sp3_interpolation_max_gap_s", 900.0)),
        sp3_epoch_match_tolerance_s=float(config.get("sp3_epoch_match_tolerance_s", 10.0)),
        attitude_nearest_max_gap_s=float(config.get("attitude_nearest_max_gap_s", 60.0)),
    )


def wavelength_for_obs(config: Mapping[str, Any], prn: str, obs_type: str) -> Optional[float]:
    system = prn[0]
    band = obs_type[1] if len(obs_type) > 1 else ""
    default_frequency_hz = GNSS_OBS_FREQUENCIES_HZ.get(system, {}).get(band)

    wavelengths = config.get("wavelengths", {})
    exact_key = f"{system}:{obs_type}"
    if exact_key in wavelengths:
        return float(wavelengths[exact_key])
    if obs_type in wavelengths:
        return float(wavelengths[obs_type])

    frequency_hz = config.get("frequencies_hz", {})
    if exact_key in frequency_hz:
        return C_MPS / float(frequency_hz[exact_key])
    if obs_type in frequency_hz:
        return C_MPS / float(frequency_hz[obs_type])
    if default_frequency_hz is not None:
        return C_MPS / default_frequency_hz
    return None


def corrected_observation_value(
    value: Optional[float],
    obs_type: str,
    delta_m: float,
    wavelength_m: Optional[float],
    correct_empty_as_zero: bool = False,
) -> Optional[float]:
    if value is None:
        if not correct_empty_as_zero:
            return None
        value = 0.0
    if obs_type.startswith(("C", "P")):
        return float(value) + delta_m
    if obs_type.startswith("L"):
        if wavelength_m is None:
            return value
        return float(value) + delta_m / wavelength_m
    return -value


def should_correct_obs(config: Mapping[str, Any], obs_type: str) -> bool:
    rules = config.get("obs_rules", {})
    prefixes = rules.get("correct_prefixes", ["C", "P", "L"])
    exclude = set(rules.get("exclude_obs_types", []))
    return obs_type not in exclude and any(obs_type.startswith(prefix) for prefix in prefixes)


def output_obs_types(config: Mapping[str, Any]) -> Optional[List[str]]:
    obs_types = config.get("output_obs_types")
    if obs_types is None:
        return None
    return list(obs_types)


def filter_output_observations(
    obs_head: Mapping[str, Any],
    obs_data: Mapping[_dt.datetime, Mapping[str, Mapping[str, Optional[float]]]],
    keep_types: Optional[List[str]],
) -> Tuple[Dict[str, Any], Dict[_dt.datetime, Dict[str, Dict[str, Optional[float]]]]]:
    if keep_types is None:
        return copy.deepcopy(obs_head), copy.deepcopy(obs_data)

    filtered_head = copy.deepcopy(obs_head)
    filtered_types = {}
    for system, band_list in filtered_head["OBS TYPES"].items():
        kept = [obs_type for obs_type in keep_types if obs_type in band_list]
        if kept:
            filtered_types[system] = kept
    filtered_head["OBS TYPES"] = filtered_types

    filtered_data: Dict[_dt.datetime, Dict[str, Dict[str, Optional[float]]]] = {}
    for epoch, epoch_data in obs_data.items():
        epoch_filtered: Dict[str, Dict[str, Optional[float]]] = {}
        for prn, observations in epoch_data.items():
            system = prn[0]
            if system == 'C' and len(prn) == 3:
                prn = 'C1' + prn[1:]
            allowed = set(filtered_types.get(system, []))
            if not allowed:
                continue
            obs_filtered = {
                obs_type: observations.get(obs_type)
                for obs_type in filtered_types[system]
                if obs_type in observations
            }
            if obs_filtered:
                epoch_filtered[prn] = obs_filtered
        if epoch_filtered:
            filtered_data[epoch] = epoch_filtered

    return filtered_head, filtered_data


def correct_epoch_satellite(
    config: Mapping[str, Any],
    context: CorrectionContext,
    antenna: Mapping[str, Any],
    epoch: _dt.datetime,
    prn: str,
    observations: Mapping[str, Optional[float]],
) -> Tuple[Dict[str, Optional[float]], List[Dict[str, Any]]]:
    key = time_key(epoch, context.time_key_decimals)
    corrected = dict(observations)
    rows: List[Dict[str, Any]] = []

    base_row = {
        "time": epoch.isoformat(sep=" "),
        "time_key": key,
        "antenna": antenna["name"],
        "prn": prn,
        "attitude_reference_frame": context.attitude_reference_frame,
        "attitude_matrix_direction": context.attitude_matrix_direction,
    }

    target_time = epoch_seconds(epoch)
    leo_state = context.leo_index.get(key)
    if leo_state is None:
        leo_state = interpolate_state(
            context.leo_series,
            target_time,
            context.sp3_interpolation_max_gap_s,
        )

    attitude = context.attitude_index.get(key)
    if attitude is None:
        attitude = nearest_attitude(
            context.attitude_times,
            context.attitude_values,
            target_time,
            context.attitude_nearest_max_gap_s,
        )

    gnss_state = context.gnss_index.get((key, prn))
    if gnss_state is None and prn in context.gnss_series:
        gnss_state = interpolate_state(
            context.gnss_series[prn],
            target_time,
            context.sp3_interpolation_max_gap_s,
        )

    if leo_state is None or attitude is None or gnss_state is None:
        reason = []
        if leo_state is None:
            reason.append("missing_leo_state")
        if attitude is None:
            reason.append("missing_attitude")
        if gnss_state is None:
            reason.append("missing_gnss_position")
        rows.append({**base_row, "status": ";".join(reason), "delta_m": np.nan})
        return corrected, rows

    try:
        rx_d_i, c_oi, _ = antenna_offset_inertial(
            antenna,
            leo_state["pos"],
            leo_state["vel"],
            attitude,
            context.attitude_matrix_direction,
            context.attitude_reference_frame,
        )
        u_nominal = line_of_sight_unit(leo_state["pos"], gnss_state["pos"])
        rx_delta_m = float(np.dot(u_nominal, rx_d_i)) * float(config.get("correction_sign", 1.0))
        ortho_error = float(np.linalg.norm(c_oi.T @ c_oi - np.eye(3)))
    except Exception as exc:
        rows.append({**base_row, "status": f"failed:{exc}", "delta_m": np.nan})
        return corrected, rows

    for obs_type, value in observations.items():
        if not should_correct_obs(config, obs_type):
            continue
        wavelength_m = wavelength_for_obs(config, prn, obs_type) if obs_type.startswith("L") else None
        if obs_type.startswith("L") and wavelength_m is None:
            rows.append(
                {
                    **base_row,
                    "obs_type": obs_type,
                    "status": "missing_wavelength",
                    "rx_delta_m": rx_delta_m,
                    "delta_m": np.nan,
                    "rx_d_i_x_m": rx_d_i[0],
                    "rx_d_i_y_m": rx_d_i[1],
                    "rx_d_i_z_m": rx_d_i[2],
                    "orbit_orthogonality_error": ortho_error,
                }
            )
            continue

        tx_pco_neu_m, tx_freq_code = gnss_pco_neu_m(context.atx_pco_index, prn, obs_type)
        try:
            tx_d_i = gnss_pco_inertial_m(
                tx_pco_neu_m,
                gnss_state["pos"],
                gnss_state["vel"],
                context.gnss_pco_frame,
            )
            gnss_apc_pos = gnss_state["pos"] + tx_d_i / 1000.0
            u = line_of_sight_unit(leo_state["pos"], gnss_apc_pos)
            tx_delta_m = float(np.dot(u, tx_d_i))
            delta_m = float(np.dot(u, rx_d_i)) * float(config.get("correction_sign", 1.0))
            rx_delta_m = delta_m
            tx_status = "tx_pco_applied" if tx_pco_neu_m is not None else "missing_tx_pco"
        except Exception as exc:
            tx_d_i = np.zeros(3)
            u = line_of_sight_unit(leo_state["pos"], gnss_state["pos"])
            tx_delta_m = np.nan
            delta_m = float(np.dot(u, rx_d_i)) * float(config.get("correction_sign", 1.0))
            rx_delta_m = delta_m
            tx_status = f"tx_pco_failed:{exc}"

        new_value = corrected_observation_value(
            value,
            obs_type,
            delta_m,
            wavelength_m,
            bool(config.get("correct_empty_observations_as_zero", False)),
        )
        corrected[obs_type] = new_value
        rows.append(
            {
                **base_row,
                "obs_type": obs_type,
                "status": "corrected" if value is not None else "empty_observation",
                "delta_m": delta_m,
                "rx_delta_m": rx_delta_m,
                "tx_delta_m": tx_delta_m,
                "tx_pco_status": tx_status,
                "tx_pco_frequency": tx_freq_code,
                "wavelength_m": wavelength_m,
                "raw_value": value,
                "corrected_value": new_value,
                "rx_d_i_x_m": rx_d_i[0],
                "rx_d_i_y_m": rx_d_i[1],
                "rx_d_i_z_m": rx_d_i[2],
                "tx_d_i_x_m": tx_d_i[0],
                "tx_d_i_y_m": tx_d_i[1],
                "tx_d_i_z_m": tx_d_i[2],
                "orbit_orthogonality_error": ortho_error,
            }
        )
    return corrected, rows


def correct_rinex_for_antenna(
    config: Mapping[str, Any],
    context: CorrectionContext,
    antenna: Mapping[str, Any],
) -> Tuple[Dict[_dt.datetime, Dict[str, Dict[str, Optional[float]]]], pd.DataFrame]:
    obs_head = Rinex.readObsHead(antenna["input_rinex"])
    obs_data = Rinex.readObs(antenna["input_rinex"], obs_head)
    if obs_head is None or obs_data is None:
        raise ValueError(f"Failed to read RINEX observation file: {antenna['input_rinex']}")

    corrected_data: Dict[_dt.datetime, Dict[str, Dict[str, Optional[float]]]] = {}
    diagnostic_rows: List[Dict[str, Any]] = []
    skipped_epoch_count = 0
    for epoch, epoch_data in obs_data.items():
        target_time = epoch_seconds(epoch)
        keep_epoch, leo_dt, gnss_dt = within_sp3_epoch_tolerance(context, target_time)
        if not keep_epoch:
            skipped_epoch_count += 1
            if config.get("diagnostic_csv"):
                diagnostic_rows.append(
                    {
                        "time": epoch.isoformat(sep=" "),
                        "time_key": time_key(epoch, context.time_key_decimals),
                        "antenna": antenna["name"],
                        "status": "dropped_epoch_sp3_time_tolerance",
                        "nearest_leo_sp3_dt_s": leo_dt,
                        "nearest_gnss_sp3_dt_s": gnss_dt,
                        "sp3_epoch_match_tolerance_s": context.sp3_epoch_match_tolerance_s,
                    }
                )
            continue

        corrected_epoch: Dict[str, Dict[str, Optional[float]]] = {}
        for prn, observations in epoch_data.items():
            corrected_obs, rows = correct_epoch_satellite(
                config,
                context,
                antenna,
                epoch,
                prn,
                observations,
            )
            corrected_epoch[prn] = corrected_obs
            diagnostic_rows.extend(rows)
        if corrected_epoch:
            corrected_data[epoch] = corrected_epoch

    print(
        f"{antenna['name']}: dropped {skipped_epoch_count} epochs by SP3 time tolerance "
        f"({context.sp3_epoch_match_tolerance_s:g} s)."
    )
    if not corrected_data:
        raise ValueError(
            f"No epochs remain for {antenna['name']} after applying SP3 time tolerance "
            f"{context.sp3_epoch_match_tolerance_s:g} s."
        )

    output_rinex = antenna["output_rinex"]
    output_dir = os.path.dirname(output_rinex)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    write_head, write_data = filter_output_observations(
        obs_head,
        corrected_data,
        output_obs_types(config),
    )
    Rinex.writeObs(write_head, write_data, output_rinex)
    return corrected_data, pd.DataFrame(diagnostic_rows)


def compute_apc_offsets(config: Mapping[str, Any]) -> pd.DataFrame:
    """Compute APC inertial offsets for all configured antennas and LEO epochs."""
    context = prepare_context(config)
    rows: List[Dict[str, Any]] = []
    for key, leo_state in context.leo_index.items():
        attitude = context.attitude_index.get(key)
        if attitude is None:
            continue
        for antenna in config["antennas"]:
            d_i, c_oi, _ = antenna_offset_inertial(
                antenna,
                leo_state["pos"],
                leo_state["vel"],
                attitude,
                context.attitude_matrix_direction,
                context.attitude_reference_frame,
            )
            rows.append(
                {
                    "time_key": key,
                    "antenna": antenna["name"],
                    "d_i_x_m": d_i[0],
                    "d_i_y_m": d_i[1],
                    "d_i_z_m": d_i[2],
                    "orbit_orthogonality_error": float(np.linalg.norm(c_oi.T @ c_oi - np.eye(3))),
                }
            )
    return pd.DataFrame(rows)


def write_diagnostics(config: Mapping[str, Any], diagnostics: List[pd.DataFrame]) -> None:
    diagnostic_csv = config.get("diagnostic_csv")
    if not diagnostic_csv:
        return
    output_dir = os.path.dirname(diagnostic_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if diagnostics:
        pd.concat(diagnostics, ignore_index=True).to_csv(diagnostic_csv, index=False)
    else:
        pd.DataFrame().to_csv(diagnostic_csv, index=False)


def main(config_path: str) -> None:
    config = load_config(config_path)
    context = prepare_context(config)
    diagnostics = []
    for antenna in config["antennas"]:
        _, diagnostic_df = correct_rinex_for_antenna(config, context, antenna)
        diagnostics.append(diagnostic_df)
        print(f"Corrected RINEX written: {antenna['output_rinex']}")
    write_diagnostics(config, diagnostics)
    if config.get("diagnostic_csv"):
        print(f"Diagnostic CSV written: {config['diagnostic_csv']}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        # print("Usage: python PCO.py PCOConfig.json")
        # sys.exit(2)
        main(r'.\PCOConfig.json')
    else:
        main(sys.argv[1])
