# DROL GNSS/POD 分析工具使用手册

## 工具目的

本工具集用于复现 DROL 任意姿态低轨卫星双天线、双系统 GNSS 精密定轨的后处理分析，覆盖：

- FINAL（双天线）、REF（单天线）和 PRE 的 SP3 轨道比较；
- RDOD 后验残差分析；
- 单天线1、单天线2和双天线融合 RINEX 的观测可用性；
- ATT-L1B 姿态四元数、姿态角速度与观测可用性的关联；
- 论文专用汇总表和英文大字号图件。

观测分析使用已修正的 `Obs_dualfreq` 和 `Obs_ant3`，不使用原始 L1B，因为其 GPS L1 载波相位的频点标记可能错误。

## 运行环境与统一入口

所有程序均在 Conda `GNSS` 环境中运行。统一入口是：

```text
D:\csu\MyTools\src\analysis\drol_analysis_cli.py
```

检查可用命令：

```powershell
conda run -n GNSS python D:\csu\MyTools\src\analysis\drol_analysis_cli.py --help
```

## 常用命令

运行全部流程：

```powershell
conda run -n GNSS python D:\csu\MyTools\src\analysis\drol_analysis_cli.py all
```

仅运行轨道与残差批处理：

```powershell
conda run -n GNSS python D:\csu\MyTools\src\analysis\drol_analysis_cli.py batch
```

指定日期：

```powershell
conda run -n GNSS python D:\csu\MyTools\src\analysis\drol_analysis_cli.py batch --dates 20241227 20241228
```

观测可用性图：

```powershell
conda run -n GNSS python D:\csu\MyTools\src\analysis\drol_analysis_cli.py observation --focus-day 20241227
```

姿态与观测关联图：

```powershell
conda run -n GNSS python D:\csu\MyTools\src\analysis\drol_analysis_cli.py attitude --attitude-days 20241226 20241227 20241228
```

仅重新生成论文对比图表：

```powershell
conda run -n GNSS python D:\csu\MyTools\src\analysis\drol_analysis_cli.py paper
```

追加 `--dry-run` 可仅显示将执行的命令。

## 数据与配置

统一配置文件为：

```text
D:\csu\MyTools\configs\DROL_AnalysisToolConfig.json
```

它分为五部分：

| JSON 节点 | 配置内容 |
|---|---|
| 顶层 `pod_root`、`log_root`、`pre_root`、`output_root` | POD、日志、PRE 与结果目录 |
| `solutions` | 每一种定轨方案的 SP3、日志目录和文件前缀 |
| `observation` | RINEX 根目录与观测重点日期 |
| `attitude` | ATT-L1B 根目录与需绘图的日期 |
| `plot` | 图件语言、字号和 DPI 的记录参数 |

例如，若增加使用 Cannonball 模型的双天线 MIX 结果，可复制一个方案并修改目录与文件前缀：

```json
"dual_antenna_mix_cannonball": {
  "pod_directory": "DROL_FINAL_MIX_CB_batch",
  "log_directory": "DROL_FINAL_MIX_CB_batch",
  "sp3_prefix": "DROL_FINAL_MIX_CB_",
  "label": "Dual-antenna GPS+BDS Cannonball"
}
```

将该对象加入 `solutions` 后执行 `batch` 即会自动扫描存在的 SP3 日期、生成残差图并对比 REF 与 PRE。若目录中没有对应的日志或 PRE 重叠，工具会保留已有结果，并在汇总中显示缺失字段。

| 数据 | 默认位置 | 用途 |
|---|---|---|
| POD SP3 | `D:\csu\GNSS_MIX\CSUAPPS\CSUPODSApp\BIN\outputdata` | FINAL/REF 和单系统结果 |
| 处理日志 | `D:\csu\GNSS_MIX\CSUAPPS\CSUPODSApp\BIN\LOG` | RDOD 残差 |
| PRE | `D:\csu\dataset\DROL_data` | 外部参考轨道 |
| 修正 RINEX | `D:\csu\dataset\DROL-GNSS\Obs_dualfreq` | Ant1/Ant2 |
| 融合 RINEX | `D:\csu\dataset\DROL-GNSS\Obs_ant3` | 双天线有效观测 |
| ATT-L1B | `D:\csu\dataset\DROL-GNSS\ATT-L1B` | 姿态四元数 |

若要使用自定义 JSON，而不是默认文件，可在任何命令后指定 `--config`：

```powershell
conda run -n GNSS python D:\csu\MyTools\src\analysis\drol_analysis_cli.py batch --config D:\my_config\DROL_custom.json
```

PRE 和 ATT 文件采用 `YYYYDDD0` 编码，例如 2024-12-27 对应 `20243620`；工具已自动完成转换。

## 输出

输出根目录：

```text
D:\csu\MyTools\analysis_output\DROL_batch
```

| 文件 | 含义 |
|---|---|
| `batch_summary.csv` | 全部方案的残差、方案间与 PRE 轨道统计 |
| `paper_orbit_comparison_summary.csv` | FINAL vs REF、REF vs PRE、FINAL vs PRE |
| `paper_dual_antenna_constellation_vs_pre.csv/png` | 双天线 GPS、BDS、GPS+BDS 与 PRE 对比 |
| `observation_epoch_availability.csv` | 逐历元 GPS/BDS/总卫星数 |
| `observation_availability_by_solution.png` | 天线方案观测组成图 |
| `low_visibility_ratio_by_solution.png` | 少于4星的历元比例 |
| `attitude_quaternion_rate_YYYYMMDD.png` | 姿态四元数和角速度 |
| `attitude_observation_relation_YYYYMMDD.png` | 姿态角速度与有效卫星数 |
| `dop_comparison_epoch_results.csv` | Ant1、Ant2、Fused 的逐历元可用卫星数与 GDOP/PDOP/TDOP/RDOP/ADOP/CDOP |
| `dop_comparison_summary.csv` | 每个方案、每天的均值、中位数和 P95 DOP |
| `dop_comparison_timeseries_YYYYMMDD.png` | 每日 GPS、BDS、GPS+BDS 三面板的 Ant1/Ant2/Fused PDOP 时间序列，图内标注 Mean/P95 |
| `force_rtn_inferred_components.png` | FORCE_RTN 四组推断 RTN 力分量时间序列 |
| `force_rtn_inferred_magnitudes.png` | FORCE_RTN 四组推断力模长的对数坐标图 |
| `force_rtn_inferred_summary.csv` | 各推断力组的均值、P95、最大模长与 RTN 均值 |

每个方案和日期目录还包含 `residual/`、`orbit_vs_single/` 和 `orbit_vs_pre/`。

## 论文解读规则

1. PRE 应称为“外部参考轨道一致性”，不是无条件的真值精度。
2. `FINAL vs REF` 是双天线与单天线的内部交叉比较；`FINAL/REF vs PRE` 是外部一致性比较。
3. 融合 RINEX 为经过连续性与去重策略筛选的**有效观测集合**，不等于 Ant1 与 Ant2 物理视场的简单并集。
4. ATT-L1B 与 RINEX 均使用 GPS 时间，可直接同步比较。
5. PRE 重叠历元较少的日期应单独标记为短弧段或特殊时段。

## 源码组成

| 脚本 | 责任 |
|---|---|
| `analysis.py` | SP3、RES 和日志解析基础模块 |
| `batch_drol_analysis.py` | JSON 驱动的轨道与残差批处理 |
| `drol_observation_quality.py` | RINEX 观测可用性 |
| `drol_attitude_analysis.py` | ATT-L1B 姿态与姿态—观测关联 |
| `export_drol_paper_comparisons.py` | 论文轨道比较表和图 |
| `drol_analysis_cli.py` | 统一入口 |
| `drol_arc_analysis.py` | 连续载波跟踪弧段分析 |
| `lib/DOPTool.py` | 可复用的 SP3 插值、RTN 变换与 DOP 几何计算函数 |
| `drol_dop_comparison.py` | Ant1、Ant2、Fused 的逐历元 DOP 对比 |
| `drol_dop_analysis.py` | 兼容入口；新分析统一使用 `drol_dop_comparison.py` |
| `drol_force_rtn_analysis.py` | 无文件头 FORCE_RTN 的推断列映射、时间转换与力学量绘图 |

## 连续弧段分析

运行命令：

```powershell
conda run -n GNSS python D:\csu\MyTools\src\analysis\drol_analysis_cli.py arcs --focus-day 20241227
```

配置节点 `arc` 中的 `gap_factor` 默认为 1.5，表示相邻历元间隔超过标称采样间隔的 1.5 倍即视为断弧。输出包括：

- `continuous_tracking_arcs.csv`：每个 PRN 的起止时间、历元数和弧段长度；
- `continuous_arc_statistics.csv`：按方案和星座统计弧段数量、均值、中位数、P95 和超过10分钟比例；
- `continuous_arc_duration_counts.png`：按时长分箱的弧段计数图；每个 Ant1、Ant2、Fused 柱由 GPS 实心段和 BDS 斜线段堆叠构成；
- `continuous_arc_timeline_YYYYMMDD.png`：重点日 PRN—时间弧段图。

## DOP 分析

配置文件的 `dop` 节点使用**显式文件列表**。公共的 `reference_orbits`、`precise_gnss_products` 分别填写 LEO 参考轨道和精密 GNSS SP3；`configurations` 填写每一种观测配置的标签、颜色和 RINEX 列表；`systems` 用 PRN 前缀定义 GPS、BDS 或组合系统。`receiver_satellite` 是 LEO 在参考 SP3 中的卫星标识，本数据集为 `L02`：

```json
"dop": {
  "receiver_satellite": "L02",
  "systems": [
    {"label": "GPS", "prefixes": ["G"]},
    {"label": "BDS", "prefixes": ["C"]},
    {"label": "GPS+BDS", "prefixes": ["G", "C"]}
  ],
  "reference_orbits": ["D:\\...\\DROL_REF_20241226.sp3"],
  "precise_gnss_products": ["D:\\...\\WUM0MGXFIN_20243610000_01D_05M_ORB.SP3"],
  "configurations": [
    {"label": "Ant1", "rinex_files": ["D:\\...\\DL0100LEO_20243610.rnx"]},
    {"label": "Ant2", "rinex_files": ["D:\\...\\DL0200LEO_20243610.rnx"]},
    {"label": "Fused", "rinex_files": ["D:\\...\\DROL_20243610_AfterPCO_Merge2Ant.rnx"]}
  ]
}
```

运行：

```powershell
conda run -n GNSS python D:\csu\MyTools\src\analysis\drol_analysis_cli.py dop
```

每个 RINEX 历元先以实际非空观测确定候选卫星；仅保留精密 GNSS SP3 和 LEO 参考轨道均可插值的卫星。少于 4 颗卫星、参考轨道无覆盖、以及几何矩阵秩亏的历元均写入 `dop_comparison_processing_status.csv`，而不会混入有效 DOP 统计。`RDOP/ADOP/CDOP` 为将位置协方差转换至参考轨道 RTN 坐标系后得到的方向 DOP，适合讨论低轨卫星不同方向上的几何强弱。

## FORCE_RTN 力学量分析

运行：

```powershell
conda run -n GNSS python D:\csu\MyTools\src\analysis\drol_analysis_cli.py forces
```

该工具将每条 `FORCE_RTN:` 记录解析为“简化儒略日 + 四组三分量”，目前按量级暂定为中心引力、非球形引力、大气阻力、太阳辐射压。MJD 时间通过 `lib/TimeSystem.py` 的 `TimeSystem.from_mjd()` 转换。由于源文件无列头，图题、CSV 字段与输出文件名均显式标记为 `inferred`；在获得生成程序的列定义前，不应将该映射写成已验证的文件格式说明。
