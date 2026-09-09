# MyGNSSTools

用于 GNSS 数据处理、DROL-GNSS 多天融合评估与结果可视化的 Python 工具集。项目将公共算法、融合流程、分析脚本和绘图接口分离，主要面向 RINEX、SP3 和 CSU POD 相关产品。

## 目录结构

| 目录 | 内容 |
| --- | --- |
| `lib/` | RINEX/SP3 读取、坐标与时间系统、天线 PCO、融合等通用库。 |
| `src/fusion/` | 双天线优先级、几何与姿态辅助的 RINEX 融合流程。 |
| `src/analysis/` | DROL POD、残差、DOP、观测质量、轨道和载波弧分析。 |
| `plot/` | 唯一的绘图实现入口，含轨道对比、后验残差和卫星数变化图。 |
| `configs/` | 分析、融合和绘图的 JSON 配置模板。 |
| `time_sync/` | GNSS 时间同步与 TDEV 评估工具。 |
| `test/` | 融合与观测质量的脚本测试。 |
| `docs/` | DROL 工作流和模块说明。 |

## 环境

推荐使用项目开发时的 Conda 环境：

```powershell
& 'C:\Users\Eren\anaconda3\envs\GNSS\python.exe' --version
```

项目依赖以科学计算与绘图库为主，例如 `numpy`、`pandas`、`matplotlib`；部分完整工作流还依赖 `astropy` 及本地 CSUPODS 运行环境。

## 常用绘图命令

所有绘图均由 JSON 配置控制。请先复制相应的 `*.example.json`，按本机数据路径调整后再运行。

```powershell
# 多 SP3 轨道产品对比：XYZ、3D 或两者；可显示 RMSE
& 'C:\Users\Eren\anaconda3\envs\GNSS\python.exe' .\plot\orbit.py `
  --config .\configs\PLotConfig\OrbitComparisonConfig.example.json

# POD 后验残差：每份数据生成时序散点图与天向图
& 'C:\Users\Eren\anaconda3\envs\GNSS\python.exe' .\plot\residual.py `
  --config .\configs\PLotConfig\PosteriorResidualConfig.example.json

# RINEX 卫星数变化
& 'C:\Users\Eren\anaconda3\envs\GNSS\python.exe' .\plot\satellite_count.py `
  --config .\configs\PLotConfig\SatelliteCountConfig.example.json
```

输入既可为单一文件、文件列表，也可为目录；绘图模块会递归寻找对应的 SP3、RES 或 RINEX 文件。组合模式由 `combine_datasets` 控制：轨道图可为每个对比组定义颜色、线形和点形；残差及卫星数对比图使用颜色区分数据集，卫星数图再以线形区分系统。详细参数见 [`plot/README.md`](plot/README.md)。

## DROL 融合与分析

融合流程入口：

```powershell
& 'C:\Users\Eren\anaconda3\envs\GNSS\python.exe' .\src\fusion\drol_fusion.py
```

多日融合 POD 批处理入口：

```powershell
& 'C:\Users\Eren\anaconda3\envs\GNSS\python.exe' .\src\analysis\drol_fusion_batch_analysis.py
```

分别通过 [`configs/DROL_FusionConfig.json`](configs/DROL_FusionConfig.json) 和 [`configs/DROL_FusionBatchAnalysis.json`](configs/DROL_FusionBatchAnalysis.json) 设置输入、输出与候选方案。完整工作流说明见 [`docs/DROL_Fusion_Toolkit.md`](docs/DROL_Fusion_Toolkit.md) 与 [`docs/DROL_GNSS_Analysis_Toolkit.md`](docs/DROL_GNSS_Analysis_Toolkit.md)。

## 数据与产物

原始观测、SP3、残差、图像、CSV 输出以及第三方 FAST 二进制工具均不纳入版本控制。请在配置中指定本机数据目录；生成结果将保留在配置所定义的输出目录。
