"""Batch SP3 and RDOD-residual analysis driven by DROL_BatchAnalysisConfig.json.

The script deliberately reuses ``analysis.py`` and ``PlotTool.py``.  It writes
one folder per day and comparison, a machine-readable summary CSV, and uses
large English fonts suitable for paper figures.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analysis
import PlotTool


def set_paper_style(font_size=16, dpi=300):
    plt.rcParams.update({
        "font.family": "Arial", "font.size": font_size, "axes.titlesize": font_size + 3,
        "axes.labelsize": font_size + 1, "xtick.labelsize": font_size - 2, "ytick.labelsize": font_size - 2,
        "legend.fontsize": font_size - 2, "figure.dpi": 160, "savefig.dpi": dpi,
        "axes.grid": True, "grid.alpha": 0.3,
    })


def dates_for(solution, pod_root: Path):
    folder = pod_root / solution["pod_directory"]
    pattern = re.compile(re.escape(solution["sp3_prefix"]) + r"(\d{8})\.sp3$")
    return {match.group(1): path for path in folder.glob("*.sp3") if (match := pattern.match(path.name))}


def pre_file(date: str, pre_root: Path):
    # CSU PRE products encode the date as YYYY + DDD + trailing product digit.
    doy = datetime.strptime(date, "%Y%m%d").strftime("%Y%j") + "0"
    found = sorted(pre_root.glob(f"REDA_{doy}_DL00.PRE"))
    return found[0] if found else None


def orbit_comparison(reference: Path, tested: Path, title: str, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "root_path": str(output), "sp3_file_1": str(reference), "sp3_file_2": str(tested),
        "newest": False, "sp3_file2_dir": "", "set_time": False, "start_time": "", "end_time": "",
        "output_path": ".", "POD_type": "orbit", "satellite": "DROL", "save_csv": True,
    }
    diff = analysis.compute_sp3_diff(config)
    PlotTool.plot_error({"analysis_path": str(output), "POD_type": "orbit", "name1": title.split(" vs ")[0],
                         "name2": title.split(" vs ")[-1], "satellite": "DROL", "savefig": True})
    return {
        "orbit_epochs": len(diff), "orbit_3d_rms_mm": float(np.sqrt(np.mean(diff["r_err_mm"] ** 2))),
        "orbit_3d_p95_mm": float(np.quantile(np.abs(diff["r_err_mm"]), 0.95)),
    }


def residual_analysis(solution_name, solution, date, log_root: Path, output: Path, mode: str, plot: bool):
    log_dir = log_root / solution["log_directory"] / f"{solution['sp3_prefix']}{date}"
    res_path = log_dir / f"DRO_{mode}_Res_{date}.dat"
    if not res_path.exists():
        return {"residual_status": "missing"}
    data = analysis.ResLog(str(log_dir), "DRO", date, mode)
    phase, code = [], []
    for satellites in data.values():
        for value in satellites.values():
            phase.append(value["prefitL"]); code.append(value["prefitC"])
    if not phase:
        return {"residual_status": "empty"}
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "residual_status": "ok", "residual_observations": len(phase),
        "phase_std_m": float(np.std(phase)), "phase_rms_m": float(np.sqrt(np.mean(np.square(phase)))),
        "code_std_m": float(np.std(code)), "code_rms_m": float(np.sqrt(np.mean(np.square(code)))),
    }
    if plot:
        plot_config = {"savefig": True, "output_path": str(output), "satellite": solution_name}
        PlotTool.visualize_res_sky(data, f"{mode}_{solution_name}_{date}", plot_config)
        plt.close("all")
    return result


def main():
    parser = argparse.ArgumentParser(description="Batch DROL SP3 and residual analysis")
    parser.add_argument("--config", type=Path, default=Path(r"D:\csu\MyTools\configs\DROL_BatchAnalysisConfig.json"))
    parser.add_argument("--dates", nargs="*", help="Optional YYYYMMDD date filter")
    parser.add_argument("--font-size", type=int, default=15)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(); set_paper_style(args.font_size, args.dpi)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    pod_root, log_root, pre_root = map(Path, (config["pod_root"], config["log_root"], config["pre_root"]))
    output_root = Path(config["output_root"]); output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "used_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    solutions = config["solutions"]; reference_name = config["reference_solution"]; reference = solutions[reference_name]
    ref_dates = dates_for(reference, pod_root)
    rows = []
    for solution_name, solution in solutions.items():
        available = dates_for(solution, pod_root)
        selected = sorted(set(available) & set(args.dates or available))
        for date in selected:
            day_output = output_root / solution_name / date
            row = {"solution": solution_name, "label": solution["label"], "date": date}
            try:
                row.update(residual_analysis(solution_name, solution, date, log_root, day_output / "residual", config["residual_mode"], config["plot_residuals"]))
            except Exception as exc:
                row["residual_status"] = f"error: {exc}"
            if config["plot_orbit_differences"] and solution_name != reference_name and date in ref_dates:
                try:
                    row.update(orbit_comparison(ref_dates[date], available[date], f"{solution_name} vs {reference_name}", day_output / "orbit_vs_single"))
                except Exception as exc:
                    row["orbit_status"] = f"error: {exc}"
            pre = pre_file(date, pre_root)
            if config["plot_orbit_differences"] and pre:
                try:
                    row.update({f"pre_{k}": v for k, v in orbit_comparison(pre, available[date], f"{solution_name} vs PRE", day_output / "orbit_vs_pre").items()})
                except Exception as exc:
                    row["pre_orbit_status"] = f"error: {exc}"
            rows.append(row); print(solution_name, date, row.get("residual_status", "not-run"))
    summary = pd.DataFrame(rows).sort_values(["solution", "date"])
    summary_path = output_root / "batch_summary.csv"
    try:
        summary.to_csv(summary_path, index=False, float_format="%.6f")
    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = output_root / f"batch_summary_{stamp}.csv"
        summary.to_csv(summary_path, index=False, float_format="%.6f")
        print(f"Existing batch_summary.csv is locked; wrote {summary_path.name} instead.")
    print(f"Saved {len(summary)} analysis records to {summary_path}")


if __name__ == "__main__":
    main()
