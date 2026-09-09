# GNSS_MIX Python layout

The `D:\csu\GNSS_MIX` top-level prototype scripts have been retired. Their
production replacements are maintained in `D:\csu\MyTools`:

| Retired purpose | Maintained entry point |
|---|---|
| Single, priority, screen, geometry, hybrid and attitude-switch RINEX fusion | `src\fusion\drol_fusion.py` with `configs\DROL_FusionConfig.json` |
| POD result, orbit, residual and observation-count evaluation | `src\analysis\drol_fusion_batch_analysis.py` with `configs\DROL_FusionBatchAnalysis.json` |
| Carrier-arc analysis | `src\analysis\drol_arc_analysis.py` |
| DROL attitude and observation visualization | `src\analysis\drol_attitude_analysis.py` |
| DROL POD / DOP analysis | `src\analysis\drol_pod_analysis.py`, `src\analysis\drol_dop_analysis.py` |

The following remains intentionally in place because it is the CSUPODS runtime
layer: `D:\csu\GNSS_MIX\CSUAPPS\CSUPODSApp\BIN\scripts`.
This includes the parallel orbit launcher, DROL configuration generators,
preprocessing utilities, GFO download utilities, and the local `MyTools`
imports required by those operational scripts.

The normal fusion command is:

```powershell
& 'C:\Users\Eren\anaconda3\envs\GNSS\python.exe' `
  'D:\csu\MyTools\src\fusion\drol_fusion.py'
```
