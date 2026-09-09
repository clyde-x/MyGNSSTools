# 历史试验脚本

本目录收纳原先直接散落在 `D:\csu\MyTools` 顶层的 PCO 试验脚本，目的是清理顶层目录并保留可追溯性。

| 文件 | 原用途 | 当前状态 |
| --- | --- | --- |
| `PCO_gemini.py` | 早期 PCO 改正实现试验 | 已由上级目录 `PCO.py` 取代 |
| `PCO_new.py` | 早期 PCO 改正实现试验 | 已由上级目录 `PCO.py` 取代 |
| `testPCO.py` | 姿态/坐标变换手工验证 | 仅供历史参考 |
| `pco_compare_epoch.py` | 单历元 PCO 输出对比 | 仅供历史参考 |
| `pco_fit_epoch.py` | 单历元 PCO 参数拟合试验 | 仅供历史参考 |

这些文件保留其历史路径假设（例如 `./lib`），不应直接作为当前处理链运行。当前流程请使用上级目录的 `run_drol_pco_batch.py`、`PCO.py` 和 `drol_fusion.py`。
