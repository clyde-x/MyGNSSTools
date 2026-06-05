import datetime as dt
from pathlib import Path

import PCO
import Rinex


EPOCH = dt.datetime(2024, 12, 27, 0, 4, 4)
TARGET = Path(r"D:\csu\GNSS_MIX\CSUAPPS\CSUPODSApp\BIN\inputdata\Obs\DRO\ANT3\DRO_3_20241227_AfterPCO_Merge3MIX_10-10.rnx")
CANDIDATE = Path(r"D:\csu\MyTools\DRO_3_20241227_FixedRTN_PCO_Test.rnx")

ORDER = {
    "G": ["C1C", "C2S", "L1C", "L2S"],
    "C": ["C2I", "C6I", "L2I", "L6I"],
}


def target_sat_id(prn):
    if prn.startswith("C1") and len(prn) == 4:
        return "C" + prn[2:]
    return prn


def parse_text_epoch(path):
    lines = path.read_text().splitlines()
    out = {}
    found = False
    remaining = 0
    for line in lines:
        if line.startswith(">"):
            parts = line.split()
            found = (
                int(parts[1]) == 2024
                and int(parts[2]) == 12
                and int(parts[3]) == 27
                and int(parts[4]) == 0
                and int(parts[5]) == 4
                and abs(float(parts[6]) - 4.0) < 1e-7
            )
            remaining = int(parts[8]) if found else 0
            continue
        if found and remaining > 0:
            prn = line[:4].strip()
            sat = target_sat_id(prn)
            pos = 4
            values = {}
            for obs_type in ORDER.get(sat[0], []):
                text = line[pos:pos + 14].strip()
                values[obs_type] = None if not text else float(text)
                pos += 16
            out[sat] = values
            remaining -= 1
            if remaining == 0:
                break
    return out


def read_candidate_epoch(path):
    head = Rinex.readObsHead(str(path))
    data = Rinex.readObs(str(path), head)
    return data[EPOCH]


target = parse_text_epoch(TARGET)
candidate = parse_text_epoch(CANDIDATE)
max_m = 0.0
count = 0
for sat in sorted(target):
    if sat not in candidate:
        print("missing", sat)
        continue
    for obs_type, target_value in target[sat].items():
        candidate_value = candidate[sat].get(obs_type)
        if target_value is None or candidate_value is None:
            continue
        if obs_type.startswith("L"):
            wl = PCO.wavelength_for_obs({}, sat, obs_type)
            diff_m = (candidate_value - target_value) * wl
            diff_value = candidate_value - target_value
            unit = "cy"
        else:
            diff_m = candidate_value - target_value
            diff_value = diff_m
            unit = "m"
        max_m = max(max_m, abs(diff_m))
        count += 1
        print(f"{sat:>3} {obs_type} diff={diff_value: .6f} {unit} ({diff_m: .6f} m)")
print(f"count={count} max_abs_m={max_m:.6f}")
