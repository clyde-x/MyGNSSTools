"""Create paper-facing summaries from the DROL batch-analysis CSV.

The comparisons are intentionally explicit:
1) FINAL (dual-antenna GPS+BDS) versus REF (single-antenna GPS+BDS);
2) REF (single-antenna GPS+BDS) versus the PRE external reference;
3) FINAL versus PRE as supporting context.
"""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(r"D:\csu\MyTools\analysis_output\DROL_batch"))
    parser.add_argument("--font-size", type=int, default=15)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    source = pd.read_csv(args.root / "batch_summary.csv", dtype={"date": str})
    final = source[source.solution.eq("dual_antenna_mix")].set_index("date")
    ref = source[source.solution.eq("single_antenna_mix")].set_index("date")
    common = final.index.intersection(ref.index).sort_values()
    table = pd.DataFrame(index=common)
    table.index.name = "date"
    table["final_vs_ref_epochs"] = final.loc[common, "orbit_epochs"]
    table["final_vs_ref_3d_rms_mm"] = final.loc[common, "orbit_3d_rms_mm"]
    table["final_vs_ref_3d_p95_mm"] = final.loc[common, "orbit_3d_p95_mm"]
    table["ref_vs_pre_epochs"] = ref.loc[common, "pre_orbit_epochs"]
    table["ref_vs_pre_3d_rms_mm"] = ref.loc[common, "pre_orbit_3d_rms_mm"]
    table["ref_vs_pre_3d_p95_mm"] = ref.loc[common, "pre_orbit_3d_p95_mm"]
    table["final_vs_pre_epochs"] = final.loc[common, "pre_orbit_epochs"]
    table["final_vs_pre_3d_rms_mm"] = final.loc[common, "pre_orbit_3d_rms_mm"]
    table["final_vs_pre_3d_p95_mm"] = final.loc[common, "pre_orbit_3d_p95_mm"]
    table.to_csv(args.root / "paper_orbit_comparison_summary.csv", float_format="%.6f")

    plt.rcParams.update({"font.family": "Arial", "font.size": args.font_size, "axes.labelsize": args.font_size + 2,
                         "axes.titlesize": args.font_size + 3, "xtick.labelsize": args.font_size - 2, "ytick.labelsize": args.font_size - 1,
                         "legend.fontsize": args.font_size - 2, "savefig.dpi": args.dpi})
    fig, ax = plt.subplots(figsize=(12, 6.5), layout="constrained")
    x = list(table.index)
    ax.plot(x, table["final_vs_ref_3d_rms_mm"], "o-", linewidth=2, label="FINAL (dual antenna) vs REF (single antenna)")
    ax.plot(x, table["ref_vs_pre_3d_rms_mm"], "s-", linewidth=2, label="REF (single antenna) vs PRE")
    ax.plot(x, table["final_vs_pre_3d_rms_mm"], "^-", linewidth=2, label="FINAL (dual antenna) vs PRE")
    ax.set_xlabel("Date")
    ax.set_ylabel("3D position difference RMS (mm)")
    ax.set_title("DROL orbit comparison against single-antenna and PRE references")
    ax.grid(True, alpha=.3)
    ax.legend()
    plt.xticks(rotation=35, ha="right")
    fig.savefig(args.root / "paper_orbit_comparison_rms.png", bbox_inches="tight")

    constellation = source[source.solution.isin(["dual_antenna_gps", "dual_antenna_bds", "dual_antenna_mix"])].copy()
    constellation = constellation[["solution", "label", "date", "pre_orbit_epochs", "pre_orbit_3d_rms_mm", "pre_orbit_3d_p95_mm"]]
    constellation = constellation.dropna(subset=["pre_orbit_3d_rms_mm"]).sort_values(["date", "solution"])
    constellation.to_csv(args.root / "paper_dual_antenna_constellation_vs_pre.csv", index=False, float_format="%.6f")
    fig, ax = plt.subplots(figsize=(12, 6.5), layout="constrained")
    display = {
        "dual_antenna_gps": "Dual antenna GPS-only vs PRE",
        "dual_antenna_bds": "Dual antenna BDS-only vs PRE",
        "dual_antenna_mix": "Dual antenna GPS+BDS vs PRE",
    }
    for solution, group in constellation.groupby("solution", sort=False):
        ax.plot(group["date"], group["pre_orbit_3d_rms_mm"], "o-", linewidth=2, label=display[solution])
    ax.set_xlabel("Date")
    ax.set_ylabel("3D position difference RMS (mm)")
    ax.set_title("Dual-antenna POD consistency with PRE: constellation comparison")
    ax.grid(True, alpha=.3)
    ax.legend()
    plt.xticks(rotation=35, ha="right")
    fig.savefig(args.root / "paper_dual_antenna_constellation_vs_pre.png", bbox_inches="tight")
    print(f"Saved {args.root / 'paper_orbit_comparison_summary.csv'}")


if __name__ == "__main__":
    main()
