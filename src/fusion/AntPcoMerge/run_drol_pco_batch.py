#!/usr/bin/env python3
"""Batch PCO correction and two-antenna merge for DROL_REF reference orbits."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path


def iter_keys(start: str, end: str):
    def to_date(key: str) -> date:
        return date(int(key[:4]), 1, 1) + timedelta(days=int(key[4:]) - 1)

    current = to_date(start)
    stop = to_date(end)
    while current <= stop:
        yield f"{current.year}{current.timetuple().tm_yday:03d}", current
        current += timedelta(days=1)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_compact_json(path: Path, value: dict):
    """Pretty-print objects while keeping short scalar lists on one line."""
    import re
    text = json.dumps(value, ensure_ascii=False, indent=2)
    # Collapse only scalar arrays; keep punctuation outside the match so an
    # existing comma after ] is preserved exactly once.
    scalar_array = re.compile(r"\[(?P<body>[^\[\]]*?)\]", re.S)
    def collapse(match):
        body = match.group("body")
        if not body.strip() or any(ch in body for ch in "{}[]"):
            return match.group(0)
        try:
            vals = json.loads("[" + body + "]")
        except json.JSONDecodeError:
            return match.group(0)
        return json.dumps(vals, ensure_ascii=False, separators=(", ", ":"))
    text = scalar_array.sub(collapse, text)
    path.write_text(text + "\n", encoding="utf-8")


def make_config(base: dict, key: str, day: date, ref_dir: Path, obs_src: Path, att_src: Path, gnss_dir: Path, out_dir: Path):
    cfg = json.loads(json.dumps(base))
    start = datetime.combine(day, datetime.min.time()).strftime("%Y-%m-%d %H:%M:%S")
    end = datetime.combine(day, datetime.max.time()).strftime("%Y-%m-%d %H:%M:%S")
    cfg["start_time"] = start
    cfg["end_time"] = end
    cfg["leo_satellite"] = "L02"
    cfg["leo_sp3_file"] = [str(ref_dir / f"DROL_REF_{day:%Y%m%d}.sp3")]
    product_days = [day - timedelta(days=1), day, day + timedelta(days=1)]
    cfg["gnss_sp3_file"] = [
        str(gnss_dir / f"WUM0MGXFIN_{d.year}{d.timetuple().tm_yday:03d}0000_01D_05M_ORB.SP3")
        for d in product_days
    ]
    cfg["attitude_file"] = str(att_src / f"DROL_{key}0.ATT")
    cfg["diagnostic_csv"] = str(out_dir / f"PCO_{key}_diagnostic.csv")
    cfg["antennas"] = [
        {
            "name": "ANT2",
            "input_rinex": str(obs_src / f"DL0200LEO_{key}0.rnx"),
            "output_rinex": str(out_dir / f"DL0200LEO_{key}0_PCO.rnx"),
            "com_to_apc_body_m": [0.7492, -0.1810, 0.1653],
            "pco": [0.7492, -0.1810, 0.1653],
        },
        {
            "name": "ANT1",
            "input_rinex": str(obs_src / f"DL0100LEO_{key}0.rnx"),
            "output_rinex": str(out_dir / f"DL0100LEO_{key}0_PCO.rnx"),
            "com_to_apc_body_m": [-0.7664, -0.1690, 0.1579],
            "pco": [-0.7664, -0.1690, 0.1579],
        },
    ]
    return cfg


def merge_one(merge_module, ant1: Path, ant2: Path, output: Path):
    header1, data1 = merge_module.read_rinex_obs(str(ant1), system="MIX")
    header2, data2 = merge_module.read_rinex_obs(str(ant2), system="MIX")
    headerout = header1.copy()
    merged, records = merge_module.merge_rinex_data_with_strategy1(data1, data2)
    output.parent.mkdir(parents=True, exist_ok=True)
    merge_module.write_merged_rinex(str(output), headerout, merged)
    return len(data1), len(data2), len(merged), len(records)


def padded_attitude(src: Path, dst: Path):
    """Extend the numeric attitude copy to the full UTC day for Slerp bounds."""
    rows = [line.strip() for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty attitude file: {src}")
    first = rows[0].split()
    last = rows[-1].split()
    first[5] = "0.00000000"
    last[5] = "59.00000000"
    rows[0] = " ".join(first)
    rows[-1] = " ".join(last)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return dst


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024359")
    parser.add_argument("--end", default="2025003")
    parser.add_argument("--keys", nargs="*")
    parser.add_argument(
        "--obs-src",
        default=r"D:\csu\dataset\DROL-GNSS\Obs_dualfreq",
        help="Directory containing the two antenna RINEX observation files.",
    )
    args = parser.parse_args()

    source_root = Path(r"D:\csu\dataset\DROL-GNSS")
    mytools = Path(r"D:\csu\MyTools")
    ref_dir = source_root / "DROL_REF"
    obs_src = Path(args.obs_src)
    att_src = source_root / "ATT-L1B"
    gnss_dir = source_root / "GNSS"
    out_dir = source_root / "Obs_pco"
    merge_dir = source_root / "Obs_ant3"
    config_path = mytools / "configs" / "PCOConfig.json"
    # Keep the complete PCO workflow in this folder.  The working directory
    # remains MyTools because PCOConfig.json and shared utility modules live
    # there.
    pco_script = Path(__file__).with_name("PCO.py")
    merge_script = mytools / "lib" / "Merge_TwoAnts_Rinex.py"
    if args.keys:
        key_set = set(args.keys)
        days = [(key, datetime.strptime(key, "%Y%j").date()) for key in args.keys]
    else:
        days = list(iter_keys(args.start, args.end))
        key_set = {key for key, _ in days}

    base = json.loads(config_path.read_text(encoding="utf-8"))
    merge_module = load_module(merge_script, "drol_merge_two_ants")
    out_dir.mkdir(parents=True, exist_ok=True)
    merge_dir.mkdir(parents=True, exist_ok=True)

    for key, day in days:
        cfg = make_config(base, key, day, ref_dir, obs_src, att_src, gnss_dir, out_dir)
        write_compact_json(config_path, cfg)
        required = [
            ref_dir / f"DROL_REF_{day:%Y%m%d}.sp3",
            obs_src / f"DL0100LEO_{key}0.rnx",
            obs_src / f"DL0200LEO_{key}0.rnx",
            att_src / f"DROL_{key}0.ATT",
            *[Path(p) for p in cfg["gnss_sp3_file"]],
        ]
        missing = [p for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing inputs for " + key + ":\n" + "\n".join(map(str, missing)))
        log_base = ref_dir / "PCO_logs"
        log_base.mkdir(parents=True, exist_ok=True)
        stdout_path = log_base / f"PCO_{key}.stdout.txt"
        stderr_path = log_base / f"PCO_{key}.stderr.txt"
        print(f"[{key}] PCO correction", flush=True)
        proc = subprocess.run(
            [sys.executable, str(pco_script)],
            cwd=str(mytools),
            text=True,
            capture_output=True,
            check=False,
        )
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(f"PCO failed for {key}, exit={proc.returncode}; see {stderr_path}")
        ant1 = out_dir / f"DL0100LEO_{key}0_PCO.rnx"
        ant2 = out_dir / f"DL0200LEO_{key}0_PCO.rnx"
        merged = merge_dir / f"DROL_{key}0_AfterPCO_Merge2Ant.rnx"
        # counts = merge_one(merge_module, ant1, ant2, merged)
        counts = merge_one(merge_module, ant2, ant1, merged)
        print(f"[{key}] merged epochs ant1={counts[0]} ant2={counts[1]} merged={counts[2]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
