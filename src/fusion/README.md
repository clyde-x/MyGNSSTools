# DROL fusion tools

All production RINEX fusion is run through `drol_fusion.py`; select methods,
dates, file templates and policy parameters in
`D:\csu\MyTools\configs\DROL_FusionConfig.json`.

```powershell
& 'C:\Users\Eren\anaconda3\envs\GNSS\python.exe' `
  'D:\csu\MyTools\src\fusion\drol_fusion.py'
```

`merge_priority.py`, `merge_dual_antenna_geometry.py`, and
`fuse_rinex_hybrid_by_attitude.py` are implementation modules used by the
unified entry point. `generate_drol_fusion_eval_configs.py` creates CSUPODS
evaluation configuration folders when needed.
