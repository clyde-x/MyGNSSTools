"""Reproducible DROL POD validation without third-party Python packages.

It reads SP3-compatible POD and PRE products, reports ECEF-to-RTN orbit
differences, and writes paper-ready SVG figures with large English text.
Coordinates in SP3/PRE position records are converted from km to m.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

UTC = timezone.utc


def vec_sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def dot(a, b): return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
def norm(a): return math.sqrt(dot(a, a))
def unit(a):
    n = norm(a)
    return (a[0]/n, a[1]/n, a[2]/n)
def cross(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def parse_sp3(path: Path):
    """Return {(time, satellite): (x_m, y_m, z_m)} from SP3 or PRE."""
    records, epoch = {}, None
    with path.open("r", encoding="ascii", errors="ignore") as f:
        for line in f:
            if line.startswith("*"):
                p = line[1:].split()
                if len(p) >= 6:
                    epoch = datetime(*map(int, p[:5]), tzinfo=UTC) + timedelta(seconds=float(p[5]))
            elif epoch and line.startswith("P"):
                sat, values = line[1:4].strip(), line[4:].split()
                if sat and len(values) >= 3:
                    xyz = tuple(float(v) for v in values[:3])
                    if max(map(abs, xyz)) < 999999.0:
                        records[(epoch, sat)] = tuple(v * 1000.0 for v in xyz)
    if not records:
        raise ValueError(f"No position data in {path}")
    return records


def read_product(directory: Path, pattern: str):
    result = {}
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {pattern} in {directory}")
    for path in files:
        result.update(parse_sp3(path))
    return result


def percentile(values, q):
    values = sorted(values)
    if not values: return float("nan")
    idx = (len(values)-1)*q
    lo, hi = int(idx), math.ceil(idx)
    return values[lo] if lo == hi else values[lo] + (values[hi]-values[lo])*(idx-lo)


def comparison(reference, test, name):
    common = sorted(set(reference) & set(test))
    # CSU PRE uses L98 whereas the POD products use L02 for the same DROL
    # spacecraft.  If both products contain exactly one target satellite,
    # align them by epoch and retain the test-product satellite label.
    if not common:
        ref_sats = {sat for _, sat in reference}
        test_sats = {sat for _, sat in test}
        if len(ref_sats) == len(test_sats) == 1:
            ref_sat, test_sat = next(iter(ref_sats)), next(iter(test_sats))
            ref_by_time = {time: xyz for (time, sat), xyz in reference.items() if sat == ref_sat}
            test_by_time = {time: xyz for (time, sat), xyz in test.items() if sat == test_sat}
            common = [(time, test_sat) for time in sorted(set(ref_by_time) & set(test_by_time))]
            reference = {(time, test_sat): xyz for time, xyz in ref_by_time.items()}
            test = {(time, test_sat): xyz for time, xyz in test_by_time.items()}
    by_sat = defaultdict(list)
    for time, sat in common:
        by_sat[sat].append((time, reference[(time, sat)], test[(time, sat)]))
    rows = []
    for sat, series in by_sat.items():
        for i, (time, ref, sol) in enumerate(series):
            before = series[max(0, i-1)]
            after = series[min(len(series)-1, i+1)]
            dt = max((after[0]-before[0]).total_seconds(), 1.0)
            velocity = tuple((after[1][j]-before[1][j])/dt for j in range(3))
            radial = unit(ref)
            normal = unit(cross(ref, velocity))
            along = cross(normal, radial)
            delta = vec_sub(sol, ref)
            rows.append({"time": time, "sat": sat, "comparison": name,
                         "r_m": dot(delta, radial), "t_m": dot(delta, along),
                         "n_m": dot(delta, normal), "d3_m": norm(delta)})
    return rows


def write_csv(rows, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["time", "sat", "comparison", "r_m", "t_m", "n_m", "d3_m"]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields); writer.writeheader()
        for row in rows:
            writer.writerow({**row, "time": row["time"].isoformat()})


def write_statistics(rows, output: Path):
    groups = defaultdict(list)
    for row in rows: groups[(row["comparison"], row["sat"])].append(row)
    fields = ["comparison", "sat", "epochs"] + [f"{x}_{m}" for x in ("r_m", "t_m", "n_m", "d3_m") for m in ("mean", "std", "rms", "p95abs", "maxabs")]
    table = []
    for (name, sat), group in sorted(groups.items()):
        entry = {"comparison": name, "sat": sat, "epochs": len(group)}
        for key in ("r_m", "t_m", "n_m", "d3_m"):
            v = [r[key] for r in group]
            entry.update({f"{key}_mean": statistics.fmean(v), f"{key}_std": statistics.stdev(v) if len(v)>1 else 0.0,
                          f"{key}_rms": math.sqrt(statistics.fmean(x*x for x in v)),
                          f"{key}_p95abs": percentile([abs(x) for x in v], .95), f"{key}_maxabs": max(map(abs, v))})
        table.append(entry)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields); writer.writeheader(); writer.writerows(table)
    return table


def svg_time_series(rows, output: Path, title: str):
    """Draw an efficient four-panel SVG plot with 18 px English labels."""
    width, height, left, right = 1400, 1200, 130, 55
    top, bottom, gap = 70, 80, 38
    panel_h = (height-top-bottom-3*gap)/4
    rows = sorted(rows, key=lambda r: r["time"])
    start, end = rows[0]["time"].timestamp(), rows[-1]["time"].timestamp()
    span = max(end-start, 1.0)
    chunks = [rows[i::max(1, len(rows)//5000)] for i in range(max(1, min(1, len(rows))))]
    # Keep all up to 5000 samples; this avoids a multi-megabyte vector file.
    sampled = rows[::max(1, math.ceil(len(rows)/5000))]
    panels = [("r_m", "Radial (m)", "#0072B2"), ("t_m", "Along-track (m)", "#D55E00"),
              ("n_m", "Cross-track (m)", "#009E73"), ("d3_m", "3D difference (m)", "#333333")]
    content = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
               '<rect width="100%" height="100%" fill="white"/>',
               f'<text x="{width/2}" y="38" text-anchor="middle" font-family="Arial" font-size="24" font-weight="bold">{title}</text>']
    for k, (key, ylabel, color) in enumerate(panels):
        y0 = top + k*(panel_h+gap); values = [r[key] for r in rows]
        lo, hi = min(values), max(values)
        margin = max((hi-lo)*.08, .01); lo -= margin; hi += margin
        def xpix(row): return left + (width-left-right)*(row["time"].timestamp()-start)/span
        def ypix(value): return y0 + panel_h - panel_h*(value-lo)/(hi-lo)
        points = " ".join(f"{xpix(r):.1f},{ypix(r[key]):.1f}" for r in sampled)
        content += [f'<line x1="{left}" y1="{y0+panel_h}" x2="{width-right}" y2="{y0+panel_h}" stroke="#333"/>',
                    f'<line x1="{left}" y1="{y0}" x2="{left}" y2="{y0+panel_h}" stroke="#333"/>',
                    f'<line x1="{left}" y1="{ypix(0):.1f}" x2="{width-right}" y2="{ypix(0):.1f}" stroke="#aaa" stroke-dasharray="5,5"/>',
                    f'<polyline fill="none" stroke="{color}" stroke-width="1.3" points="{points}"/>',
                    f'<text x="24" y="{y0+panel_h/2}" font-family="Arial" font-size="18" transform="rotate(-90 24 {y0+panel_h/2})">{ylabel}</text>',
                    f'<text x="{left-10}" y="{y0+16}" text-anchor="end" font-family="Arial" font-size="16">{hi:.2f}</text>',
                    f'<text x="{left-10}" y="{y0+panel_h}" text-anchor="end" font-family="Arial" font-size="16">{lo:.2f}</text>']
    content += [f'<text x="{left}" y="{height-35}" font-family="Arial" font-size="18">{rows[0]["time"].strftime("%Y-%m-%d %H:%M")}</text>',
                f'<text x="{width-right}" y="{height-35}" text-anchor="end" font-family="Arial" font-size="18">{rows[-1]["time"].strftime("%Y-%m-%d %H:%M UTC")}</text>', '</svg>']
    output.write_text("\n".join(content), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="DROL SP3/PRE RTN validation")
    p.add_argument("--pod-root", type=Path, default=Path(r"D:\csu\GNSS_MIX\CSUAPPS\CSUPODSApp\BIN\outputdata"))
    p.add_argument("--pre-root", type=Path, default=Path(r"D:\csu\dataset\DROL_data"))
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "drol_analysis_output")
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    single = read_product(args.pod_root / "DROL_REF_MIX_batch", "*.sp3")
    dual = read_product(args.pod_root / "DROL_FINAL_MIX_batch", "*.sp3")
    pre = read_product(args.pre_root, "REDA_*_DL00.PRE")
    rows = comparison(single, dual, "dual_minus_single") + comparison(pre, single, "single_minus_pre") + comparison(pre, dual, "dual_minus_pre")
    write_csv(rows, args.output / "orbit_differences_rtn.csv")
    table = write_statistics(rows, args.output / "orbit_statistics.csv")
    for name in sorted({r["comparison"] for r in rows}):
        svg_time_series([r for r in rows if r["comparison"] == name], args.output / f"rtn_timeseries_{name}.svg", f"DROL POD validation: {name.replace('_', ' ')}")
    print(f"Saved {len(rows):,} matched epochs to {args.output}")
    for item in table: print(item["comparison"], item["sat"], f"3D RMS={item['d3_m_rms']:.3f} m", f"95%={item['d3_m_p95abs']:.3f} m")


if __name__ == "__main__": main()
