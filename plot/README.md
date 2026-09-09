# 绘图模块

`plot` 是项目唯一的绘图实现入口。分析脚本负责读取、计算和导出表格；本目录的函数只接收已经计算好的数据并输出图像。

| 模块 | 用途 | 推荐接口 |
| --- | --- | --- |
| `plot.drol` | DROL 批处理和论文图 | `plot_observation_availability`、`plot_force_rtn`、`plot_frequency_counts` |
| `plot.orbit` | 多 SP3 轨道产品对比 | `plot_sp3_orbit_comparison` |
| `plot.residual` | POD 后验残差时序与天向图 | `plot_posterior_residuals` |
| `plot.satellite_count` | RINEX 卫星数时间序列 | `plot_satellite_counts_from_config` |
| `plot.filter` | 滤波、残差、周跳与卫星可见性诊断 | `plot_state`、`plot_clk_bias`、`plot_lambda_fix_analysis`、`visualize_sats` |
| `plot.legacy` | 原 `PlotTool` 的 POD/SPP/残差图接口 | `plot_error`、`visualize_res_sky`、`visualize_dyn` |

```python
from plot.drol import plot_observation_availability

path = plot_observation_availability(epoch_table, output_dir, "20241228")
```

## 多 SP3 轨道对比

推荐通过 JSON 配置运行：

```powershell
& 'C:\Users\Eren\anaconda3\envs\GNSS\python.exe' .\plot\orbit.py `
  --config .\configs\PLotConfig\OrbitComparisonConfig.example.json
```

模板见 [`configs/PLotConfig/OrbitComparisonConfig.example.json`](../configs/PLotConfig/OrbitComparisonConfig.example.json)。配置中的相对路径以配置文件所在文件夹为基准；`reference_files` 和每个 `folder` 都可以是文件夹、单个 SP3 文件或文件列表。

```python
from plot.orbit import plot_sp3_orbit_comparison

result = plot_sp3_orbit_comparison(
    reference_files=r"D:\data\DROL_REF",  # 文件夹：自动递归读取全部 .sp3
    comparisons=[
        (r"D:\data\DROL_SINGLE", "Single antenna"),
        (r"D:\data\DROL_FINAL", "Dual antenna"),
    ],
    satellite="L02",  # 多星 SP3 时指定；单星 SP3 可省略
    title="Orbit error against reference",
    kind="plot",                         # 或 "scatter"
    view="both",                          # "xyz"、"3d" 或 "both"
    scatter_size=3.0,                     # scatter 默认点面积；可被 style 的 s 覆盖
    trim_percent=1.0,                     # 仅隐藏每组最大的 1% 3D 误差
    start_time="2024-12-28 00:00:00",    # 可省略
    end_time="2024-12-28 23:59:59",      # 可省略
    output_path="figures/orbit_compare.png",
    styles={
        "Single antenna": {"color": "#D55E00", "marker": "o", "linestyle": "--", "lw": 1.2},
        "Dual antenna": {"color": "#0072B2", "marker": "^", "linestyle": "-", "lw": 1.2},
    },
)
print(result.metrics)  # x/y/z 与 3D RMSE、共同历元数
```

每个输入既可传单个文件、文件夹，也可传文件/文件夹的列表；文件夹会**递归**读取其下全部 `.sp3` 文件。若每个输入组各自都只有一颗卫星，`satellite` 参数（以及配置中的该字段）可省略，程序会自动选择它，即使参考轨道和待比轨道的卫星号不同（例如 `L98` 对 `L02`）。多星 SP3 才必须指定卫星号；若各组 ID 不同，可配置为 `"satellite": {"reference": "L98", "ant1": "L02"}`。`view` 可选 `"xyz"`、`"3d"` 或 `"both"`；`trim_percent` 是每组按 3D 误差隐藏的最大误差百分比（只影响显示，不影响 RMSE）。`scatter_size` 默认是更紧凑的 `3.0`，各组可用 `style.s` 覆盖。每个 `style` 可用 `marker` 设置形状，如 `"o"`（圆）、`"s"`（方）、`"^"`（上三角）、`"D"`（菱形）、`"P"`（十字）。`comparisons` 也可以直接传字典，例如 `{"Single": "single_folder", "Final": ["part1.sp3", "part2.sp3"]}`。散点图的样式可用 `{"Final": {"color": "#0072B2", "marker": "o", "s": 2, "alpha": .7}}`。

旧的 `lib.PlotTool` 与 `src.visualization.NewFilterVisualize` 是兼容转发模块；新代码应直接从 `plot` 导入。

## 后验残差图

`plot_posterior_residuals` 对每份残差数据固定输出两张图：`*_posterior_time.png`（相位/伪距时序散点）与 `*_posterior_sky.png`（相位/伪距北向上天向图）。推荐使用独立配置 [`configs/PLotConfig/PosteriorResidualConfig.example.json`](../configs/PLotConfig/PosteriorResidualConfig.example.json)：它只含残差文件/文件夹输入、输出目录和绘图参数，不依赖 `tempdata_root`、`outputdata_root` 等融合或定轨路径。`time_clip_percent` 会针对相位与伪距分别隐藏时序散点中绝对值最大的对应百分比；默认模板为 `5.0`，设置为 `0.0` 即不截取。将 `combine_datasets` 设为 `true` 时，所有数据集的相位/伪距散点会叠加到 `posterior_residual_time_comparison.png`，便于横向比较。

```powershell
& 'C:\Users\Eren\anaconda3\envs\GNSS\python.exe' .\plot\residual.py `
  --config .\configs\PLotConfig\PosteriorResidualConfig.example.json
```

## RINEX 卫星数变化图

[`configs/PLotConfig/SatelliteCountConfig.example.json`](../configs/PLotConfig/SatelliteCountConfig.example.json) 支持单个 RINEX、多个文件或文件夹（递归读取），并可在 `systems` 中选择 `G`、`C`、`R`、`E`、`J`、`S`、`I` 和 `ALL`。每个数据集输出卫星数 PNG、逐历元 CSV 和均值/最小/最大值汇总 CSV；将 `combine_datasets` 设为 `true` 时，所有数据集/系统曲线会输出到同一张 `satellite_count_comparison.png`：颜色区分数据集、线形区分系统。可通过 `dataset_colors` 为数据集指定固定颜色。

```powershell
& 'C:\Users\Eren\anaconda3\envs\GNSS\python.exe' .\plot\satellite_count.py `
  --config .\configs\PLotConfig\SatelliteCountConfig.example.json
```
