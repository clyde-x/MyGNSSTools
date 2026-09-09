"""Regression and diagnostic tests for Merge_TwoAnts_Rinex.py.

Run:
    py test_merge_two_ants_rinex.py
    py test_merge_two_ants_rinex.py --primary <primary.rnx> --secondary <secondary.rnx>

The script only extracts the two target functions from the production source, so it
does not require matplotlib, numpy, or pandas just to test the merge logic.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path


DEFAULT_PRIMARY = Path(r"D:\csu\dataset\DROL-GNSS\Obs_pco\DL0200LEO_20243600_PCO.rnx")
DEFAULT_SECONDARY = Path(r"D:\csu\dataset\DROL-GNSS\Obs_pco\DL0100LEO_20243600_PCO.rnx")
TARGET = Path(r"D:\csu\MyTools\lib\Merge_TwoAnts_Rinex.py")


def load_target_functions():
    """Load exactly the current production implementations without its GUI imports."""
    tree = ast.parse(TARGET.read_text(encoding="utf-8"))
    wanted = {"get_satellite_segments", "merge_rinex_data_with_strategy1"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if {node.name for node in nodes} != wanted:
        raise RuntimeError(f"Could not find {wanted} in {TARGET}")
    namespace = {"defaultdict": defaultdict, "timedelta": timedelta}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TARGET), "exec"), namespace)
    return namespace["get_satellite_segments"], namespace["merge_rinex_data_with_strategy1"]


def epoch(second: int):
    return datetime(2024, 1, 1, 0, 0, second)


def make_data(mapping):
    return {
        time: {"time": time, "flag": "0", "num_satellites": len(sats),
               "observations": {sat: {"C1C": 1.0} for sat in sats}}
        for time, sats in mapping.items()
    }


def test_continuous_arc(get_segments):
    data = make_data({epoch(0): ["G01"], epoch(10): ["G01"], epoch(20): ["G01"]})
    segments = get_segments(data)
    actual = [(item["start_time"], item["end_time"], item["duration"]) for item in segments["G01"]]
    expected = [(epoch(0), epoch(20), 20.0)]
    return actual == expected, f"continuous arc: expected {expected}, got {actual}"


def test_primary_priority(merge):
    primary = make_data({epoch(0): ["G01"], epoch(10): ["G01"]})
    secondary = make_data({epoch(0): ["G01", "C01"], epoch(10): ["G01", "C01"]})
    merged, records = merge(primary, secondary, min_continuous_time=0)
    source = records["G01"][epoch(0)]["data_source"]
    sats = set(merged[epoch(0)]["observations"])
    ok = source == "data1" and sats == {"G01", "C01"}
    return ok, f"primary priority: G01 source={source}; merged satellites={sorted(sats)}"


def read_rinex_satellites(path: Path):
    """Read just epoch/satellite IDs from RINEX 3, sufficient for count diagnostics."""
    data, current, in_body = {}, None, False
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not in_body:
                in_body = "END OF HEADER" in line
                continue
            if line.startswith(">"):
                fields = line[1:].split()
                year, month, day, hour, minute = map(int, fields[:5])
                seconds = float(fields[5])
                current = datetime(year, month, day, hour, minute) + timedelta(seconds=seconds)
                data[current] = {"time": current, "flag": "0", "num_satellites": 0, "observations": {}}
            elif current is not None and len(line) >= 3 and line[0].isalpha():
                sat = line[:3].strip()
                if len(sat) == 3:
                    data[current]["observations"][sat] = line.rstrip("\n")
    for value in data.values():
        value["num_satellites"] = len(value["observations"])
    return data


def diagnose_files(merge, primary_path, secondary_path):
    primary, secondary = read_rinex_satellites(primary_path), read_rinex_satellites(secondary_path)
    merged, records = merge(primary, secondary)
    totals, problem_epochs = Counter(), []
    for time in sorted(set(primary) | set(secondary)):
        p = set(primary.get(time, {}).get("observations", {}))
        s = set(secondary.get(time, {}).get("observations", {}))
        m = set(merged[time]["observations"])
        totals.update(primary=len(p), secondary=len(s), union=len(p | s), merged=len(m),
                      supplemented=len(m - p), missing_from_union=len((p | s) - m))
        totals["merged_less_than_primary_epochs"] += len(m) < len(p)
        totals["merged_less_than_secondary_epochs"] += len(m) < len(s)
        totals["merged_not_greater_than_primary_epochs"] += len(m) <= len(p)
        totals["merged_not_greater_than_secondary_epochs"] += len(m) <= len(s)
        if m != p | s:
            problem_epochs.append((time, len(p), len(s), len(m), len(p | s), sorted((p | s) - m)))
    sources = Counter(item["data_source"] for per_sat in records.values() for item in per_sat.values())
    print(f"Files: primary={primary_path.name}, secondary={secondary_path.name}")
    print(f"Epochs: {len(merged)}; total satellite observations: primary={totals['primary']}, "
          f"secondary={totals['secondary']}, union={totals['union']}, merged={totals['merged']}")
    print(f"Secondary observations actually selected: {totals['supplemented']}; "
          f"missing relative to per-epoch union: {totals['missing_from_union']}")
    print("Epoch comparisons: "
          f"merged < primary={totals['merged_less_than_primary_epochs']}, "
          f"merged < secondary={totals['merged_less_than_secondary_epochs']}, "
          f"merged <= primary={totals['merged_not_greater_than_primary_epochs']}, "
          f"merged <= secondary={totals['merged_not_greater_than_secondary_epochs']}")
    print(f"Selection records: {dict(sources)}; epochs not equal to union: {len(problem_epochs)}")
    for row in problem_epochs[:5]:
        print(f"  {row[0]} primary={row[1]} secondary={row[2]} merged={row[3]} union={row[4]} missing={row[5]}")
    return not problem_epochs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--secondary", type=Path, default=DEFAULT_SECONDARY)
    args = parser.parse_args()
    get_segments, merge = load_target_functions()
    tests = [test_continuous_arc(get_segments), test_primary_priority(merge)]
    for passed, detail in tests:
        print(("PASS" if passed else "FAIL") + " - " + detail)
    if args.primary.exists() and args.secondary.exists():
        diagnose_files(merge, args.primary, args.secondary)
    else:
        print("RINEX sample files not found; synthetic tests completed only.")


if __name__ == "__main__":
    main()
