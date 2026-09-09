# DROL Fusion Toolkit

The public fusion entry point is `D:\csu\MyTools\src\fusion\drol_fusion.py`.
All normal fusion runs are controlled by
`D:\csu\MyTools\configs\DROL_FusionConfig.json`; no per-method command
line needs to be assembled.

```powershell
& 'C:\Users\Eren\anaconda3\envs\GNSS\python.exe' `
  'D:\csu\MyTools\src\fusion\drol_fusion.py'
```

Use `--dry-run` to validate the selected input/output paths without writing
RINEX files. `--config <path>` is available only for an alternate JSON file.

## Fusion configuration

`date_range`, `inputs`, and `output.pattern` define the common run scope.
Paths support `{date}` (`YYYYMMDD`), `{tag}` (`YYYYDDD0`) and `{method}`.
Enable a method by setting its `enabled` field to `true`.

| `method` | Meaning | Main parameters |
|---|---|---|
| `single_ant1` / `single_ant2` | Copy either antenna as the single-antenna reference | none |
| `priority` | Antenna-2-priority union; fills complementary valid arcs from Antenna 1 | `min_continuous_time` |
| `geometry` | Per-satellite selection using geometry score | `model`, `hysteresis_margin` |
| `geometry_hysteresis` | Geometry selection with switching hysteresis | `model`, `hysteresis_margin` |
| `geometry_screen` | Geometry selection with score screening | `model`, `screen_threshold` |
| `hybrid` | Attitude switching plus a side-view fusion window | `side_window_deg`, `min_arc_points` |
| `attitude_switch` | Pure attitude switching (`hybrid` with no side window) | `min_arc_points` |

`model` is a three-value list: `[observation_type, zenith_weight, elevation_weight]`.
The supplied config uses PCO-corrected observations in
`D:\csu\dataset\DROL-GNSS\Obs_pco` and writes each strategy to its own
subdirectory below `D:\csu\dataset\DROL-GNSS\fusion`.

The older policy modules in `src` are retained as tested implementation
libraries. Do not invoke them for routine production fusion; select the method
in the JSON config instead.

## POD analysis

```powershell
& 'C:\Users\Eren\anaconda3\envs\GNSS\python.exe' `
  'D:\csu\MyTools\src\analysis\drol_fusion_batch_analysis.py'
```

Its configuration is `D:\csu\MyTools\configs\DROL_FusionBatchAnalysis.json`.
It produces JSON and PNG files only; it does not create CSV files.

`run_gfo_daily_orbits_parallel.py` remains in
`D:\csu\GNSS_MIX\CSUAPPS\CSUPODSApp\BIN\scripts`, since it is the CSUPODS
runtime launcher rather than a fusion or analysis utility.
