"""Fuse PCO-corrected dual-antenna RINEX observations using viewing geometry.

For each common satellite observation, the antenna whose boresight has the
larger dot product with the satellite line-of-sight (LOS) in the spacecraft
body frame is retained.  PCO has already been applied to the inputs, so the
selected data share the spacecraft COM reference point.

The script also provides a ``priority`` baseline compatible with the previous
Ant2-first merge.  ``geometry_hysteresis`` prevents rapid receiver switching
when the two boresight scores are nearly equal, while still filling gaps with
the other antenna.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

_EPOCH_TRANSFORMS = {}
_LEO_POSITIONS = {}

ROOT = Path(__file__).resolve().parents[1]
MYTOOLS = Path(r"D:\csu\MyTools")
sys.path.insert(0, str(MYTOOLS / "lib"))
import Rinex  # noqa: E402
import Sp3Tools  # noqa: E402
import TimeSystem  # noqa: E402
import MathTool  # noqa: E402

ANTENNA_BORESIGHTS = {
    "ant1": np.array([-1.0, 0.0, 0.0]),  # -X
    "ant2": np.array([1.0, 0.0, 0.0]),   # +X
}


def as_timesystem(epoch: datetime):
    result = TimeSystem.TimeSystem()
    result._update_from_datetime(epoch)
    return result


def prepare_sp3(files):
    frame = Sp3Tools.read_sp3([str(path) for path in files])
    result = {}
    for sat, group in frame.groupby("sat_id"):
        group = group.drop_duplicates("timesystem").sort_values("timesystem")
        time0 = group["timesystem"].iloc[0]
        result[sat] = (time0, np.array([time - time0 for time in group["timesystem"]]),
                       group[["x", "y", "z"]].to_numpy(float) * 1000.0)
    return result


def interpolate_position(records, time):
    time0, times, xyz = records
    target = time - time0
    if target < times[0] or target > times[-1]:
        return None
    return np.array([MathTool.window_interpolate(times, xyz[:, axis], target)[0]
                     for axis in range(3)])


def prepare_attitude(path: Path):
    rows = []
    for line in path.open(encoding="utf-8", errors="ignore"):
        fields = line.split()
        if len(fields) != 10:
            continue
        try:
            year, month, day, hour, minute = map(int, fields[:5])
            second = float(fields[5]); quat = list(map(float, fields[6:10]))
        except ValueError:
            continue
        rows.append((datetime(year, month, day, hour, minute, int(second),
                              round((second % 1) * 1e6)), quat))
    if not rows:
        raise ValueError(f"No attitude rows found in {path}")
    time0 = rows[0][0]
    times = np.array([(time - time0).total_seconds() for time, _ in rows])
    return time0, times, Slerp(times, Rotation.from_quat(np.array([q for _, q in rows])))


def body_los(epoch, gnss_id, leo_sp3, gnss_sp3, attitude):
    epoch_ts = as_timesystem(epoch)
    leo_xyz = _LEO_POSITIONS.get(epoch)
    if leo_xyz is None:
        leo_xyz = interpolate_position(leo_sp3["L02"], epoch_ts)
        _LEO_POSITIONS[epoch] = leo_xyz
    if leo_xyz is None or gnss_id not in gnss_sp3:
        return None
    gnss_xyz = interpolate_position(gnss_sp3[gnss_id], epoch_ts)
    if gnss_xyz is None:
        return None
    inertial_to_body_ecef = _EPOCH_TRANSFORMS.get(epoch)
    if inertial_to_body_ecef is None:
        time0, attitude_times, slerp = attitude
        sec = float(np.clip((epoch - time0).total_seconds(), attitude_times[0], attitude_times[-1]))
        # ATT quaternion represents inertial -> body; Astropy supplies inertial -> ECEF.
        inertial_to_body = slerp([sec]).as_matrix()[0]
        # A GMST-only inertial-to-ECEF rotation is adequate for an antenna
        # hemisphere test (the sub-arcsecond EOP terms do not affect it) and
        # is fast enough to evaluate every 10-s observation epoch.
        utc = epoch - timedelta(seconds=18)
        days = (utc - datetime(2000, 1, 1, 12)).total_seconds() / 86400.0
        theta = np.deg2rad((280.46061837 + 360.98564736629 * days) % 360.0)
        c, s = np.cos(theta), np.sin(theta)
        inertial_to_ecef = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
        inertial_to_body_ecef = inertial_to_body @ inertial_to_ecef.T
        _EPOCH_TRANSFORMS[epoch] = inertial_to_body_ecef
    los_ecef = gnss_xyz - leo_xyz
    los_ecef /= np.linalg.norm(los_ecef)
    return inertial_to_body_ecef @ los_ecef


def valid_membership(data, minimum_arc_seconds):
    membership = defaultdict(set)
    per_sat = defaultdict(list)
    all_epochs = sorted(data)
    interval = np.median([(b - a).total_seconds() for a, b in zip(all_epochs, all_epochs[1:]) if b > a])
    for epoch, sats in data.items():
        for sat in sats:
            per_sat[sat].append(epoch)
    for sat, epochs in per_sat.items():
        begin = 0
        for index in range(1, len(epochs) + 1):
            if index != len(epochs) and (epochs[index] - epochs[index - 1]).total_seconds() <= interval * 1.5:
                continue
            if (epochs[index - 1] - epochs[begin]).total_seconds() + interval >= minimum_arc_seconds:
                membership[sat].update(epochs[begin:index])
            begin = index
    return membership


def select_observations(data1, data2, policy, geometry, min_arc, hysteresis, minimum_score):
    valid1, valid2 = valid_membership(data1, min_arc), valid_membership(data2, min_arc)
    last = {}
    selected, diagnostics = {}, []
    for epoch in sorted(set(data1) | set(data2)):
        output = {}
        for sat in sorted(set(data1.get(epoch, {})) | set(data2.get(epoch, {}))):
            has1 = sat in data1.get(epoch, {}) and epoch in valid1.get(sat, set())
            has2 = sat in data2.get(epoch, {}) and epoch in valid2.get(sat, set())
            if not (has1 or has2):
                continue
            score1 = score2 = None
            source = "ant1" if has1 else "ant2"
            reason = "only_ant1" if has1 else "only_ant2"
            if has1 and has2:
                if policy == "priority":
                    source, reason = "ant2", "priority_ant2"
                else:
                    los = body_los(epoch, sat, *geometry)
                    if los is None:
                        source, reason = "ant2", "geometry_unavailable_ant2"
                    else:
                        score1 = float(np.dot(ANTENNA_BORESIGHTS["ant1"], los))
                        score2 = float(np.dot(ANTENNA_BORESIGHTS["ant2"], los))
                        source = "ant1" if score1 >= score2 else "ant2"
                        reason = "geometry"
                        if policy == "geometry_hysteresis" and sat in last and abs(score1 - score2) < hysteresis:
                            source, reason = last[sat], "geometry_hysteresis"
            # The two receiver streams are almost disjoint in this dataset, so
            # geometry is chiefly valuable as a quality gate for single-source
            # observations, not merely as a tie-breaker for duplicates.
            if policy == "geometry_screen":
                los = body_los(epoch, sat, *geometry)
                if los is not None:
                    source_score = float(np.dot(ANTENNA_BORESIGHTS[source], los))
                    if source_score < minimum_score:
                        diagnostics.append({"epoch": epoch.isoformat(sep=" "), "satellite": sat, "source": source,
                                            "reason": "screened_low_score", "score_ant1": score1 if score1 is not None else (source_score if source == "ant1" else None),
                                            "score_ant2": score2 if score2 is not None else (source_score if source == "ant2" else None)})
                        continue
                    if reason.startswith("only_"):
                        reason = "geometry_screen_pass"
                        if source == "ant1": score1 = source_score
                        else: score2 = source_score
            output[sat] = data1[epoch][sat] if source == "ant1" else data2[epoch][sat]
            last[sat] = source
            diagnostics.append({"epoch": epoch.isoformat(sep=" "), "satellite": sat, "source": source,
                                "reason": reason, "score_ant1": score1, "score_ant2": score2})
        if output:
            selected[epoch] = output
    return selected, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ant1", type=Path, required=True)
    parser.add_argument("--ant2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", choices=["priority", "geometry", "geometry_hysteresis", "geometry_screen"], default="geometry_hysteresis")
    parser.add_argument("--attitude", type=Path, required=True)
    parser.add_argument("--leo-sp3", type=Path, nargs="+", required=True)
    parser.add_argument("--gnss-sp3", type=Path, nargs="+", required=True)
    parser.add_argument("--minimum-arc-seconds", type=float, default=50.0)
    parser.add_argument("--hysteresis", type=float, default=0.15)
    parser.add_argument("--minimum-score", type=float, default=0.0,
                        help="Minimum antenna-boresight/LOS cosine for geometry_screen.")
    args = parser.parse_args()
    header = Rinex.readObsHead(str(args.ant1))
    data1 = Rinex.readObs(str(args.ant1), header)
    data2 = Rinex.readObs(str(args.ant2), Rinex.readObsHead(str(args.ant2)))
    geometry = (prepare_sp3(args.leo_sp3), prepare_sp3(args.gnss_sp3), prepare_attitude(args.attitude))
    merged, diagnostics = select_observations(data1, data2, args.policy, geometry,
                                                args.minimum_arc_seconds, args.hysteresis, args.minimum_score)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Rinex.writeObs(header, merged, str(args.output))
    counts = Counter(row["source"] for row in diagnostics)
    reasons = Counter(row["reason"] for row in diagnostics)
    print(f"policy={args.policy} epochs={len(merged)} observations={len(diagnostics)} "
          f"ant1={counts['ant1']} ant2={counts['ant2']} reasons={dict(reasons)}")


if __name__ == "__main__":
    main()
