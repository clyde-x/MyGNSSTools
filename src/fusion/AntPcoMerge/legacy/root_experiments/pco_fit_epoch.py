import copy
import datetime as dt
import json
from pathlib import Path

import numpy as np
import sys
sys.path.insert(0, './lib')

import PCO
import Rinex


EPOCH = dt.datetime(2024, 12, 27, 0, 4, 4)
RAW = Path(r"D:\csu\GNSS_MIX\CSUAPPS\CSUPODSApp\BIN\inputdata\Obs\DRO\ANT3\DRO_3_20241227_L0_Merge_20-20.rnx")
TARGET = Path(r"D:\csu\GNSS_MIX\CSUAPPS\CSUPODSApp\BIN\inputdata\Obs\DRO\ANT3\DRO_3_20241227_AfterPCO_Merge3MIX_10-10.rnx")

TARGET_ORDER = {
    "G": ["C1C", "C2S", "L1C", "L2S"],
    "C": ["C2I", "C6I", "L2I", "L6I"],
}


def target_sat_id(prn):
    if prn.startswith("C1") and len(prn) == 4:
        return "C" + prn[2:]
    return prn


def parse_target_epoch(path):
    lines = path.read_text().splitlines()
    out = {}
    found = False
    remaining = 0
    for line in lines:
        if line.startswith(">"):
            parts = line.split()
            found = parts[:7] == [">", "2024", "12", "27", "0", "4", "4.0000000"]
            remaining = int(parts[8]) if found else 0
            continue
        if found and remaining > 0:
            prn = line[:4].strip()
            sys = prn[0]
            sat = target_sat_id(prn)
            vals = {}
            pos = 4
            for obs_type in TARGET_ORDER.get(sys, []):
                text = line[pos:pos + 14].strip()
                vals[obs_type] = None if not text else float(text)
                pos += 16
            out[sat] = vals
            remaining -= 1
            if remaining == 0:
                break
    return out


def raw_epoch(path):
    head = Rinex.readObsHead(str(path))
    data = Rinex.readObs(str(path), head)
    return data[EPOCH]


def observed_deltas(raw, target):
    rows = []
    for sat, target_obs in target.items():
        if sat not in raw:
            continue
        raw_obs = raw[sat]
        for obs_type, target_value in target_obs.items():
            raw_value = raw_obs.get(obs_type)
            if target_value is None or raw_value is None:
                continue
            if obs_type.startswith(("C", "P")):
                delta = target_value - raw_value
            elif obs_type.startswith("L"):
                wl = PCO.wavelength_for_obs({}, sat, obs_type)
                delta = (target_value - raw_value) * wl
            else:
                continue
            rows.append((sat, obs_type, delta))
    return rows


def predicted_delta(context, ant, sat, obs_type, sign, include_tx):
    key = PCO.time_key(EPOCH, context.time_key_decimals)
    target_time = PCO.epoch_seconds(EPOCH)
    leo_state = context.leo_index.get(key) or PCO.interpolate_state(
        context.leo_series, target_time, context.sp3_interpolation_max_gap_s
    )
    attitude = context.attitude_index.get(key) or PCO.nearest_attitude(
        context.attitude_times, context.attitude_values, target_time, context.attitude_nearest_max_gap_s
    )
    gnss_state = context.gnss_index.get((key, sat)) or PCO.interpolate_state(
        context.gnss_series[sat], target_time, context.sp3_interpolation_max_gap_s
    )
    rx_d_i, _, _ = PCO.apc_offset_inertial(
        ant["com_to_apc_body_m"],
        leo_state["pos"],
        leo_state["vel"],
        attitude,
        context.attitude_matrix_direction,
        context.attitude_reference_frame,
    )
    tx_neu, _ = PCO.gnss_pco_neu_m(context.atx_pco_index, sat, obs_type)
    tx_d_i = PCO.gnss_pco_inertial_m(tx_neu, gnss_state["pos"], gnss_state["vel"], context.gnss_pco_frame)
    u = PCO.line_of_sight_unit(leo_state["pos"], gnss_state["pos"] + tx_d_i / 1000.0)
    delta = float(np.dot(u, rx_d_i))
    if include_tx:
        delta += float(np.dot(u, tx_d_i))
    return sign * delta


def main():
    base = json.loads(Path("./configs/PCOConfig.json").read_text())
    raw = raw_epoch(RAW)
    target = parse_target_epoch(TARGET)
    obs = observed_deltas(raw, target)
    print("observed count", len(obs), "satellites", sorted({r[0] for r in obs}))

    refs = [("inertial", "inertial_to_body"), ("inertial", "body_to_inertial"),
            ("orbit", "orbit_to_body"), ("orbit", "body_to_orbit")]
    results = []
    for ref, direction in refs:
        for offset in [18.0, 0.0, -18.0]:
            cfg = copy.deepcopy(base)
            cfg["attitude_reference_frame"] = ref
            cfg["attitude_matrix_direction"] = direction
            cfg["attitude_time_offset_s"] = offset
            context = PCO.prepare_context(cfg)
            for ant in cfg["antennas"]:
                for sign in [1.0, -1.0]:
                    for include_tx in [False, True]:
                        residuals = []
                        for sat, obs_type, measured in obs:
                            try:
                                pred = predicted_delta(context, ant, sat, obs_type, sign, include_tx)
                            except Exception:
                                continue
                            residuals.append(pred - measured)
                        if residuals:
                            arr = np.asarray(residuals)
                            results.append((float(np.sqrt(np.mean(arr * arr))), float(np.mean(arr)),
                                            ref, direction, offset, ant["name"], sign, include_tx, len(arr)))
    for row in sorted(results)[:20]:
        print("rms={:.4f} mean={:.4f} ref={} dir={} off={:+.0f} ant={} sign={:+.0f} tx={} n={}".format(*row))

    best = sorted(results)[0]
    _, _, ref, direction, offset, ant_name, sign, include_tx, _ = best
    cfg = copy.deepcopy(base)
    cfg["attitude_reference_frame"] = ref
    cfg["attitude_matrix_direction"] = direction
    cfg["attitude_time_offset_s"] = offset
    context = PCO.prepare_context(cfg)
    ant = next(a for a in cfg["antennas"] if a["name"] == ant_name)
    print("best detail")
    for sat, obs_type, measured in obs:
        if obs_type.startswith("L"):
            continue
        pred = predicted_delta(context, ant, sat, obs_type, sign, include_tx)
        print(sat, obs_type, "measured", f"{measured:.4f}", "pred", f"{pred:.4f}", "res", f"{pred-measured:.4f}")

    key = PCO.time_key(EPOCH, context.time_key_decimals)
    target_time = PCO.epoch_seconds(EPOCH)
    leo_state = context.leo_index.get(key) or PCO.interpolate_state(
        context.leo_series, target_time, context.sp3_interpolation_max_gap_s
    )
    c_oi, _ = PCO.orbit_frame_matrices(leo_state["pos"], leo_state["vel"])
    a = []
    y = []
    for sat, obs_type, measured in obs:
        if not obs_type.startswith(("C", "P")):
            continue
        gnss_state = context.gnss_index.get((key, sat)) or PCO.interpolate_state(
            context.gnss_series[sat], target_time, context.sp3_interpolation_max_gap_s
        )
        u_i = PCO.line_of_sight_unit(leo_state["pos"], gnss_state["pos"])
        a.append(c_oi @ u_i)
        y.append(measured)
    a = np.asarray(a)
    y = np.asarray(y)
    d_o, *_ = np.linalg.lstsq(a, y, rcond=None)
    residual = a @ d_o - y
    print("least squares orbit-frame d_o", d_o, "rms", float(np.sqrt(np.mean(residual * residual))))
    for candidate in base["antennas"]:
        print("config", candidate["name"], "com_to_apc_body_m", candidate["com_to_apc_body_m"])


if __name__ == "__main__":
    main()
