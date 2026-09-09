"""Unified command-line entry point for the DROL POD analysis toolkit.

Run this script with the Conda ``GNSS`` environment.  It intentionally starts
the existing task-specific programs in separate processes so every command
uses the same interpreter and retains its own reproducible configuration.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = {
    "batch": "batch_drol_analysis.py",
    "observation": "drol_observation_quality.py",
    "arcs": "drol_arc_analysis.py",
    "attitude": "drol_attitude_analysis.py",
    "paper": "export_drol_paper_comparisons.py",
    "dop": "drol_dop_comparison.py",
    "l1b-frequencies": "drol_l1b_frequency_analysis.py",
    "forces": "drol_force_rtn_analysis.py",
}


def run(script: str, extra: list[str], dry_run: bool) -> None:
    command = [sys.executable, str(ROOT / SCRIPTS[script]), *extra]
    print("\n[RUN] " + " ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DROL POD analysis toolkit. Run with: conda run -n GNSS python drol_analysis_cli.py <command>")
    parser.add_argument("command", choices=["batch", "observation", "arcs", "attitude", "paper", "dop", "l1b-frequencies", "forces", "all"],
                        help="Analysis task to run")
    parser.add_argument("--config", type=Path, default=Path(r"D:\csu\MyTools\configs\DROL_AnalysisToolConfig.json"),
                        help="Unified JSON configuration file")
    parser.add_argument("--dates", nargs="*", help="Optional YYYYMMDD dates for batch processing")
    parser.add_argument("--focus-day", help="Focus day for the observation time-series plot; overrides JSON")
    parser.add_argument("--attitude-days", nargs="*", help="YYYYMMDD dates for attitude plots; overrides JSON")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    plot = config.get("plot", {})
    plot_args = ["--font-size", str(plot.get("font_size", 15)), "--dpi", str(plot.get("save_dpi", 300))]
    commands = [args.command] if args.command != "all" else ["batch", "observation", "arcs", "attitude", "paper", "dop"]
    for command in commands:
        extra: list[str] = []
        if command == "batch":
            extra += ["--config", str(args.config), *plot_args]
            if args.dates:
                extra += ["--dates", *args.dates]
        elif command == "observation":
            obs = config["observation"]
            extra += ["--data-root", obs["data_root"], "--output", config["output_root"],
                      "--focus-day", args.focus_day or obs["focus_day"], *plot_args]
        elif command == "arcs":
            arc = config["arc"]
            extra += ["--data-root", arc["data_root"], "--output", config["output_root"],
                      "--focus-day", args.focus_day or arc["focus_day"], "--gap-factor", str(arc["gap_factor"]), *plot_args]
        elif command == "attitude":
            att = config["attitude"]
            extra += ["--attitude-root", att["data_root"], "--observation-csv", str(Path(config["output_root"]) / "observation_epoch_availability.csv"),
                      "--output", config["output_root"], "--days", *(args.attitude_days or att["days"])]
            extra += plot_args
        elif command == "paper":
            extra += ["--root", config["output_root"], *plot_args]
        elif command == "dop":
            extra += ["--config", str(args.config), *plot_args]
        elif command == "l1b-frequencies":
            l1b = config["l1b_frequency"]
            extra += ["--data-root", l1b["data_root"], "--output", config["output_root"],
                      "--focus-day", args.focus_day or l1b["focus_day"], *plot_args]
        elif command == "forces":
            force = config["force_rtn"]
            extra += ["--input", force["input_file"], "--output", config["output_root"],
                      "--max-points", str(force.get("max_points", 12000)), *plot_args]
        run(command, extra, args.dry_run)


if __name__ == "__main__":
    main()
