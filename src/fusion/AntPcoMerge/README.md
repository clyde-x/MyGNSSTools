# AntPcoMerge

用于 DROL 双天线 GNSS 观测的 **PCO 改正 → 融合 RINEX 生成 → CSUPODS 配置生成** 的集中工具目录。

## 处理链与边界

```text
Obs_dualfreq（两副原始/双频筛选观测）
  → PCO.py
Obs_pco（两副 PCO 后观测）
  → drol_fusion.py
fusion/<candidate>（单天线或双天线融合观测）
  → generate_drol_fusion_eval_configs.py
CSUPODS Config/<candidate>_batch
```

PCO 已在观测文件中施加后，POD 的接收机 PCO 不能再次施加；否则会发生双重改正。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `PCO.py` | 核心 PCO 改正。以天线相位中心在星体坐标系中的向量为输入，结合 ATT、LEO SP3 与 GNSS SP3 计算视线投影改正，并写回伪距和载波观测。 |
| `run_drol_pco_batch.py` | 多日 PCO 批处理驱动。定义 Ant1/Ant2 的 PCO 向量，写入逐日 `PCOConfig.json`，调用同目录 `PCO.py`，并输出 `Obs_pco`。 |
| `drol_fusion.py` | 唯一的配置驱动融合入口。读取 JSON 的 `methods` 列表并逐日执行。 |
| `merge_priority.py` | 并集/主天线优先策略。Ant2 优先，Ant1 在主天线无有效观测时补充。 |
| `merge_dual_antenna_geometry.py` | 几何选择、带滞回选择和前视半球筛选策略。 |
| `fuse_rinex_hybrid_by_attitude.py` | 姿态驱动整天线切换与侧视窗混合策略。 |
| `generate_drol_fusion_eval_configs.py` | 从已验证模板生成隔离的 CSUPODS 两阶段配置。 |
| `DROL_FusionConfig.example.json` | 全部融合方案的可编辑示例配置。 |

## 融合方法标识

| `method` | 含义 | 主要参数 |
| --- | --- | --- |
| `single_ant1` / `single_ant2` | 单天线基线 | 无 |
| `priority` | 主天线优先的有效弧段并集 | `minimum_arc_seconds` |
| `geometry` | 逐星几何优选 | `minimum_arc_seconds` |
| `geometry_hysteresis` | 几何优选并抑制频繁切换 | `hysteresis` |
| `geometry_screen` | 有效前视半球筛选后融合 | `minimum_score`、`minimum_arc_seconds` |
| `hybrid` | 姿态切换加侧视窗内逐星补充 | `side_window_deg`、`minimum_switch_minutes` |
| `attitude_switch` | 纯姿态整天线切换 | `minimum_switch_minutes` |

## 常用命令

先在 `DROL_FusionConfig.example.json` 中核对日期、输入和输出路径，然后执行：

```powershell
# 仅检查文件与计划，不写输出
python .\drol_fusion.py --config .\DROL_FusionConfig.example.json --dry-run

# 执行已启用的融合方案
python .\drol_fusion.py --config .\DROL_FusionConfig.example.json

# 对原始双频数据批量生成两副 PCO 后观测
python .\run_drol_pco_batch.py --start 2024359 --end 2025003
```

运行 PCO 批处理前，请在 `run_drol_pco_batch.py` 的 `make_config()` 中复核 Ant1/Ant2 PCO 向量、LEO 参考 SP3、姿态文件和 GNSS 精密星历路径。融合步骤的输入应为 `Obs_pco`，不是未改正的 `Obs_dualfreq`。

## 兼容性

原有 `D:\csu\MyTools\lib\PCO.py` 与 `D:\csu\MyTools\src\fusion\` 下的同名脚本未删除，避免已有自动化命令失效。今后新增或修改功能应以本目录为准，并在验证后同步兼容入口，避免两套实现漂移。

顶层历史 PCO 试验脚本已迁入 `legacy/root_experiments/`；其中内容仅用于回溯，不属于当前处理链。
