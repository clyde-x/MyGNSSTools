"""Attitude-driven DROL dual-antenna fusion with a side-view hybrid window.

Outside the side-view window, copy one complete receiver epoch according to
which ±X antenna is nearer the local zenith.  Inside the window, retain the
union and choose the longer continuous carrier arc when both receivers track
the same PRN.  Inputs must already be PCO-corrected.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
MYTOOLS = Path(r"D:\csu\MyTools")
sys.path.insert(0, str(MYTOOLS / "lib"))
import Rinex  # noqa: E402

from merge_dual_antenna_geometry import as_timesystem, interpolate_position, prepare_attitude, prepare_sp3


def body_from_ecef(epoch, attitude):
    """ECEF -> body matrix, with the same quaternion transpose as PCO.py."""
    time0, att_times, slerp = attitude
    sec = float(np.clip((epoch - time0).total_seconds(), att_times[0], att_times[-1]))
    c_b_i = slerp([sec]).as_matrix()[0].T  # J2K -> body, PCO convention
    utc = epoch - timedelta(seconds=18)
    days = (utc - datetime(2000, 1, 1, 12)).total_seconds() / 86400.0
    theta = np.deg2rad((280.46061837 + 360.98564736629 * days) % 360.0)
    c, s = np.cos(theta), np.sin(theta)
    c_e_i = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    return c_b_i @ c_e_i.T


def leo_position(epoch, leo_sp3):
    return interpolate_position(leo_sp3["L02"], as_timesystem(epoch))


def attitude_row(epoch, leo_sp3, attitude):
    position = leo_position(epoch, leo_sp3)
    if position is None:
        return None
    c_b_e = body_from_ecef(epoch, attitude)
    radial = position / np.linalg.norm(position)  # local zenith in ECEF
    zenith_body = c_b_e @ radial
    score_ant2 = float(zenith_body[0])
    theta2 = float(np.degrees(np.arccos(np.clip(score_ant2, -1, 1))))
    # A numerical velocity is sufficient for a local LVLH orientation report.
    before, after = leo_position(epoch - timedelta(seconds=10), leo_sp3), leo_position(epoch + timedelta(seconds=10), leo_sp3)
    euler = [np.nan, np.nan, np.nan]
    if before is not None and after is not None:
        normal = np.cross(position, after - before); normal /= np.linalg.norm(normal)
        along = np.cross(normal, radial)
        c_e_l = np.column_stack((radial, along, normal))
        euler = Rotation.from_matrix(c_b_e @ c_e_l).as_euler("xyz", degrees=True).tolist()
    return {"epoch": epoch, "zenith_dot_ant2": score_ant2, "theta_ant2_deg": theta2,
            "theta_ant1_deg": 180.0 - theta2, "roll_deg": euler[0], "pitch_deg": euler[1], "yaw_deg": euler[2]}


def smooth_minimum_runs(labels, epochs, minimum_seconds):
    """Merge a short switch into the longer adjacent label, repeatedly."""
    labels = labels.copy(); changed = True
    while changed:
        changed = False; start = 0
        for end in range(1, len(labels) + 1):
            if end < len(labels) and labels[end] == labels[start]:
                continue
            duration = (epochs[end - 1] - epochs[start]).total_seconds() + 10
            if duration < minimum_seconds and start > 0 and end < len(labels):
                left, right = labels[start - 1], labels[end]
                left_begin, right_end = start - 1, end
                while left_begin > 0 and labels[left_begin - 1] == left: left_begin -= 1
                while right_end < len(labels) - 1 and labels[right_end + 1] == right: right_end += 1
                left_duration = (epochs[start - 1] - epochs[left_begin]).total_seconds() + 10
                right_duration = (epochs[right_end] - epochs[end]).total_seconds() + 10
                replacement = left if left == right or left_duration >= right_duration else right
                labels[start:end] = [replacement] * (end - start)
                changed = True; break
            start = end
    return labels


def carrier_arc_lengths(data):
    """Return continuous tracking duration per (epoch, PRN); gap > 15 s breaks."""
    result = defaultdict(dict); per_sat = defaultdict(list)
    for epoch, sats in data.items():
        for sat, values in sats.items():
            if any(value not in (None, "", 0.0) for key, value in values.items() if key.startswith("L")):
                per_sat[sat].append(epoch)
    for sat, times in per_sat.items():
        start = 0
        for end in range(1, len(times) + 1):
            if end < len(times) and (times[end] - times[end - 1]).total_seconds() <= 15:
                continue
            duration = (times[end - 1] - times[start]).total_seconds() + 10
            for epoch in times[start:end]: result[epoch][sat] = duration
            start = end
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ant1", type=Path, required=True); p.add_argument("--ant2", type=Path, required=True)
    p.add_argument("--attitude", type=Path, required=True); p.add_argument("--leo-sp3", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["switch", "hybrid"], default="hybrid")
    p.add_argument("--side-window-deg", type=float, default=15.0)
    p.add_argument("--minimum-switch-minutes", type=float, default=20.0)
    a = p.parse_args()
    h1, h2 = Rinex.readObsHead(str(a.ant1)), Rinex.readObsHead(str(a.ant2))
    d1, d2 = Rinex.readObs(str(a.ant1), h1), Rinex.readObs(str(a.ant2), h2)
    epochs = sorted(set(d1) | set(d2)); leo, att = prepare_sp3(a.leo_sp3), prepare_attitude(a.attitude)
    rows = [attitude_row(epoch, leo, att) for epoch in epochs]; rows = [row for row in rows if row is not None]
    valid_epochs = [row["epoch"] for row in rows]
    labels = ["ant2" if row["theta_ant2_deg"] <= row["theta_ant1_deg"] else "ant1" for row in rows]
    labels = smooth_minimum_runs(labels, valid_epochs, a.minimum_switch_minutes * 60)
    schedule = {epoch: label for epoch, label in zip(valid_epochs, labels)}
    limit = np.sin(np.deg2rad(a.side_window_deg))
    for row in rows:
        row["selected_antenna"] = schedule[row["epoch"]]
        row["hybrid_window"] = abs(row["zenith_dot_ant2"]) <= limit
    arcs1, arcs2 = carrier_arc_lengths(d1), carrier_arc_lengths(d2)
    merged = {}
    for row in rows:
        epoch, chosen = row["epoch"], row["selected_antenna"]
        if a.mode == "switch" or not row["hybrid_window"]:
            source = d1 if chosen == "ant1" else d2
            if epoch in source: merged[epoch] = source[epoch].copy()
            continue
        output = {}
        for sat in sorted(set(d1.get(epoch, {})) | set(d2.get(epoch, {}))):
            one, two = sat in d1.get(epoch, {}), sat in d2.get(epoch, {})
            if one and two:
                source = "ant1" if arcs1[epoch].get(sat, 0) >= arcs2[epoch].get(sat, 0) else "ant2"; reason = "longer_carrier_arc"
            else: source, reason = ("ant1", "only_ant1") if one else ("ant2", "only_ant2")
            output[sat] = d1[epoch][sat] if source == "ant1" else d2[epoch][sat]
        if output: merged[epoch] = output
    a.output.parent.mkdir(parents=True, exist_ok=True); Rinex.writeObs(h1, merged, str(a.output))
    print(f"epochs={len(merged)} observations={sum(map(len, merged.values()))} output={a.output}")


if __name__ == "__main__": main()
