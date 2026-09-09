# DROL analysis tools

## Multi-day fusion POD analysis

`drol_fusion_batch_analysis.py` is the single configuration-driven entry point
for comparing DROL antenna/fusion POD results across multiple days. It reads
the CSUPODS products and logs, then produces figures and machine-readable
outputs without requiring per-day commands.

Run it in the GNSS environment:

```powershell
& 'C:\Users\Eren\anaconda3\envs\GNSS\python.exe' `
  'D:\csu\MyTools\src\analysis\drol_fusion_batch_analysis.py'
```

The default configuration is
`D:\csu\MyTools\configs\DROL_FusionBatchAnalysis.json`. Change the date
range, candidate batches, paths, and `plots.pre_epoch_rms_ymax_cm` there. The
latter limits only the displayed 3D PRE RMS/error points; it does not discard
any data from the CSV.

`raw_rinex_pattern` gives the default raw-fusion RINEX naming rule. A solution
may override it with its own `raw_rinex_pattern`; this is used for the Ant1 and
Ant2 PCO-corrected input files.

### Required inputs per candidate and day

- `outputdata/<batch>/<prefix><YYYYMMDD>.sp3`: estimated orbit.
- `LOG/<batch>/<prefix><YYYYMMDD>/DRO_RDOD_Res_<YYYYMMDD>.dat`: RDOD residuals.
- `TempData/<batch>/<prefix><YYYYMMDD>/DRO_DataEditing_Obs_<YYYYMMDD>.rnx`:
  DataEditing output.
- External `REDA_<YYYYDOY>0_DL00.PRE`: reference orbit, when available.

### Outputs

All outputs are written to `paths.output_root` in the configuration.

| File | Content |
|---|---|
| `pod_batch_summary.json` | Per-day status/metrics and multi-day summary. |
| `pre_3d_epoch_rms_all_dates.csv` | Every common estimated/reference-orbit epoch. `shown_in_clipped_plot` records whether its value is within the configured plotting threshold. |
| `rdod_residuals_all_dates.csv` | Every RDOD observation used by the residual plots: candidate, epoch, satellite, elevation, azimuth, phase (`prefitL`) and code (`prefitC`) residuals. |
| `raw_fusion_satellite_counts_by_epoch.csv` | Raw fused-RINEX satellite counts: one epoch per row and one solution per column; unavailable epochs are blank. |
| `dataediting_satellite_counts_by_epoch.csv` | DataEditing-RINEX satellite counts in the same epoch-row/solution-column layout. |
| `observation_satellite_counts_all_dates.csv` | Long-form source table for the two wide satellite-count CSV files. |
| `pre_3d_epoch_rms_scatter_all_dates.png` | Multi-day, per-epoch 3D PRE RMS/error scatter. Values above the configured threshold are omitted only visually. |
| `rdod_phase_rms_by_date.png` | Daily phase-residual RMS overview. |
| `observation_satellite_counts_all_dates.png` | Two-panel multi-day satellite-count time series before and after DataEditing. |
| `residual_multiday/RDOD_<candidate>_all_dates_res.png` | One four-panel multi-day residual figure per candidate: phase/code time series and phase/code sky plots. |

The two CSV files use UTF-8 with BOM so they open directly in Excel. Units are
recorded in column names: orbit RMS is metres and centimetres; residuals are
metres; elevation and azimuth are degrees.

### DataEditing GPS/BDS availability figure

To reuse the three-panel availability figure (total, GPS, and BDS counts) for
the current POD candidates after DataEditing, run:

```powershell
& 'C:\Users\Eren\anaconda3\envs\GNSS\python.exe' `
  'D:\csu\MyTools\src\analysis\drol_observation_quality.py' `
  --fusion-batch-config 'D:\csu\MyTools\configs\DROL_FusionBatchAnalysis.json' `
  --output 'D:\csu\article\DROL\merge\analysis\pod_batch' `
  --focus-day 20241227 --solutions single_ant1 single_ant2 priority
```

This produces `dataediting_observation_timeseries_<date>.png` and two CSV
files containing the epoch-level GPS/BDS/total counts and per-day summaries.

## Other DROL analyses

The remaining `drol_*.py` modules are focused analyses or helpers. The older
dispatcher remains available when its broader workflow is needed:

```powershell
& 'C:\Users\Eren\anaconda3\envs\GNSS\python.exe' `
  'D:\csu\MyTools\src\analysis\drol_analysis_cli.py' all
```
