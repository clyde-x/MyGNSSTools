import json 
import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
import re
import io

import Sp3Tools
import TimeSystem
import PlotTool
import Rinex
import SetCommonSats

import Fast.plot.plotSatNum as Fplot_satnum

def len0(x):
    if x is None:
        return 0
    elif isinstance(x, (list, set, dict)):
        if len(x) == 0:
            return 0
        return len(x)
    else:
        return 0
    
def get_outlier_indices(data, droprate):
    """
    找出偏离均值(0)最大的前 droprate 比例的数据索引
    :param data: 列表 / numpy数组 / pandas.Series
    :param droprate: 剔除比例 0~1，例如 0.1 表示剔除最大的10%
    :return: 异常点的索引数组
    """
    arr = np.asarray(data)
    abs_deviation = np.abs(arr)

    n_drop = int(len(arr) * droprate)
    if n_drop <= 0:
        return np.array([], dtype=int)

    outlier_indices = np.argpartition(abs_deviation, -n_drop)[-n_drop:]
    outlier_indices = np.sort(outlier_indices)
    return outlier_indices

def split_kbr_windows(df, time_gap_threshold=30.0, residual_step_threshold=None,
                      residual_step_sigma=8.0, min_step_threshold=0.02):
    """
    Split matched KBR epochs into continuous windows before estimating bias.

    A new window starts when either:
    - gps_time is discontinuous by more than time_gap_threshold seconds
    - raw_diff has an adjacent step larger than residual_step_threshold

    If residual_step_threshold is None, a robust threshold is estimated from
    adjacent raw_diff changes, with min_step_threshold as the lower bound.
    """
    if len(df) == 0:
        return pd.Series(dtype=int), 0, residual_step_threshold

    time_diff = df['gps_time'].diff()
    time_break = time_diff > time_gap_threshold

    raw_step = df['raw_diff'].diff().abs()
    if residual_step_threshold is None:
        finite_steps = raw_step.replace([np.inf, -np.inf], np.nan).dropna()
        if len(finite_steps) > 0:
            median_step = finite_steps.median()
            mad = (finite_steps - median_step).abs().median()
            robust_sigma = 1.4826 * mad
            auto_threshold = median_step + residual_step_sigma * robust_sigma
            residual_step_threshold = max(float(auto_threshold), min_step_threshold)
        else:
            residual_step_threshold = min_step_threshold

    residual_break = raw_step > residual_step_threshold
    window_id = (time_break.fillna(False) | residual_break.fillna(False)).cumsum()

    return window_id.astype(int), int(window_id.max()) + 1, residual_step_threshold

def plot_sat_num_diff(obs_path_a, obs_path_b, tag=''):
    # 读取观测数据
    obs_head_a = Rinex.readObsHead(obs_path_a)
    obs_data_a = Rinex.readObs(obs_path_a, obs_head_a)
    
    obs_head_b = Rinex.readObsHead(obs_path_b)
    obs_data_b = Rinex.readObs(obs_path_b, obs_head_b)
    
    # 获取共视卫星列表
    sat_A = SetCommonSats.get_common_sats(obs_data_a, obs_data_a)
    sat_B = SetCommonSats.get_common_sats(obs_data_b, obs_data_b)
    common_sats_dict = SetCommonSats.get_common_sats(obs_data_a, obs_data_b)
    
    # 绘制卫星数量差异图
    Fplot_satnum.plotSatNum(sat_A, pngFile=f'sat_num_A{tag}.png')
    Fplot_satnum.plotSatNum(sat_B, pngFile=f'sat_num_B{tag}.png')
    Fplot_satnum.plotSatNum(common_sats_dict, pngFile=f'common_sats_diff{tag}.png')

    df_common = pd.DataFrame(common_sats_dict.items(), columns=['Epoch', 'Common_Satellites'])
    df_common['Num_Common_Sats'] = df_common['Common_Satellites'].apply(len0)

    df_A = pd.DataFrame(sat_A.items(), columns=['Epoch', 'Satellites_A'])
    df_A['Num_Sats_A'] = df_A['Satellites_A'].apply(len0)
    df_B = pd.DataFrame(sat_B.items(), columns=['Epoch', 'Satellites_B'])
    df_B['Num_Sats_B'] = df_B['Satellites_B'].apply(len0)

    df = pd.merge(df_A[['Epoch', 'Num_Sats_A']], df_B[['Epoch', 'Num_Sats_B']], on='Epoch', how='outer')
    df = pd.merge(df, df_common[['Epoch', 'Num_Common_Sats']], on='Epoch', how='outer')

    for i in range(len(df)):
        for col in ['Num_Sats_A', 'Num_Sats_B', 'Num_Common_Sats']:
            if pd.isna(df.at[i, col]):
                df.at[i, col] = 0
    df.to_csv(f'sat_num_diff{tag}.csv', index=False)


def read_kbr(file_path):
    header_lines = []
    data_lines = []
    is_header = True
    
    # 1. 读取文件，根据 End of YAML header 拆分
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if is_header:
                header_lines.append(line)
                if "# End of YAML header" in line:
                    is_header = False
            else:
                data_lines.append(line)
    
    # 2. 使用正则表达式从 YAML 块中提取变量名
    # 匹配模式为: - 变量名:
    col_names = []
    in_variables_block = False
    for line in header_lines:
        if "variables:" in line:
            in_variables_block = True
            continue
        if in_variables_block:
            # 匹配形如 "- gps_time:" 的行
            match = re.search(r'-\s+([a-zA-Z0-9_]+):', line)
            if match:
                col_names.append(match.group(1))
    
    # 3. 将数据块加载到 DataFrame
    data_str = "".join(data_lines)
    df = pd.read_csv(io.StringIO(data_str), sep='\s+', names=col_names, header=None)
    
    return df

def process_kbr(df):
    kbr_epoch = TimeSystem.TimeSystem().from_ymdhms(2000, 1, 1, 12, 0, 0)

    # 质量控制: 确保 qualflg 为 0 (无相位跳变或插值异常)
    df['qualflg'] = df['qualflg'].astype(int)  
    df_clean = df[df['qualflg'] == 0].copy()

    df_clean['kbr_true_range'] = (
        df_clean['biased_range'] + 
        df_clean['lighttime_corr'] + 
        df_clean['ant_centr_corr'] + 
        df_clean['iono_corr']
    )

    df_clean['timesystem'] = df_clean['gps_time'].apply(lambda seconds: kbr_epoch + float(seconds))
    cols = ['timesystem'] + [c for c in df_clean.columns if c != 'timesystem']
    df_clean = df_clean[cols]

    return df_clean


def calculate_kbr_residuals(df_sp3_c, df_sp3_d, df_kbr, starttime=None, endtime=None,
                            time_gap_threshold=30.0, residual_step_threshold=None,
                            residual_step_sigma=8.0, min_step_threshold=0.02):
    if starttime is not None:
        df_sp3_c = df_sp3_c[df_sp3_c['timesystem'] >= starttime]
        df_sp3_d = df_sp3_d[df_sp3_d['timesystem'] >= starttime]
        df_kbr = df_kbr[df_kbr['timesystem'] >= starttime]
    if endtime is not None:
        df_sp3_c = df_sp3_c[df_sp3_c['timesystem'] <= endtime]
        df_sp3_d = df_sp3_d[df_sp3_d['timesystem'] <= endtime]
        df_kbr = df_kbr[df_kbr['timesystem'] <= endtime]
    
    # 1. 提取所需列，并确保坐标单位是米 (假设若 X < 100000，则原单位是公里)
    df_c = df_sp3_c[['gps_time', 'x', 'y', 'z']].copy()
    df_d = df_sp3_d[['gps_time', 'x', 'y', 'z']].copy()
    
    if abs(df_c['x'].iloc[0]) < 100000:
        for col in ['x', 'y', 'z']:
            df_c[col] *= 1000.0
            df_d[col] *= 1000.0
            
    # 重命名列名以便合并，防止 X,Y,Z 冲突
    df_c.columns = ['gps_time', 'x_c', 'y_c', 'z_c']
    df_d.columns = ['gps_time', 'x_d', 'y_d', 'z_d']

    # 2. 核心操作：利用 inner join 对齐时间
    # 这一步会自动剔除所有不在 SP3 时间点上的 KBR 数据，以及缺失的历元
    df_merged = pd.merge(df_c, df_d, on='gps_time', how='inner')
    df_merged = pd.merge(df_merged, df_kbr[['gps_time', 'kbr_true_range']], on='gps_time', how='inner')
    df_merged = df_merged.sort_values('gps_time').reset_index(drop=True)

    if len(df_merged) == 0:
        print("警告：未找到匹配的公共历元！请检查数据的时间段是否重合。")
        return None, None

    # 3. 计算 GNSS 几何基线长度 (欧氏距离)
    df_merged['gnss_range'] = np.sqrt(
        (df_merged['x_d'] - df_merged['x_c'])**2 +
        (df_merged['y_d'] - df_merged['y_c'])**2 +
        (df_merged['z_d'] - df_merged['z_c'])**2
    )

    # 4. 去均值处理 (Detrend)
    # 剥离 KBR 观测量中包含的浮点模糊度偏置
    df_merged['raw_diff'] = df_merged['gnss_range'] - df_merged['kbr_true_range']
    window_id, window_count, used_step_threshold = split_kbr_windows(
        df_merged,
        time_gap_threshold=time_gap_threshold,
        residual_step_threshold=residual_step_threshold,
        residual_step_sigma=residual_step_sigma,
        min_step_threshold=min_step_threshold
    )
    df_merged['window_id'] = window_id
    df_merged['window_bias_m'] = df_merged.groupby('window_id')['raw_diff'].transform('mean')
    df_merged['residual'] = df_merged['raw_diff'] - df_merged['window_bias_m']
    mean_bias = df_merged['raw_diff'].mean()
    
    # 换算为厘米 (cm)
    df_merged['residual_cm'] = df_merged['residual'] * 100.0

    # 5. 计算最终统计指标
    rms = np.sqrt(np.mean(df_merged['residual_cm']**2))
    std = np.std(df_merged['residual_cm'])
    window_stats = (
        df_merged.groupby('window_id')
        .agg(
            start_gps_time=('gps_time', 'first'),
            end_gps_time=('gps_time', 'last'),
            points=('gps_time', 'size'),
            mean_bias_m=('window_bias_m', 'first')
        )
        .reset_index()
    )
    
    stats = {
        'Points_Matched': len(df_merged),
        'Mean_Bias_m': mean_bias,
        'Windows': window_count,
        'Time_Gap_Threshold_s': time_gap_threshold,
        'Residual_Step_Threshold_m': used_step_threshold,
        'Window_Biases': window_stats.to_dict('records'),
        'Residual_RMS_cm': rms,
        'Residual_STD_cm': std
    }
    
    print(f"验证完成！共严格匹配 {stats['Points_Matched']} 个同历元数据点。")
    print(f"GNSS-KBR 距离残差 RMS (去均值后): {rms:.3f} cm")
    
    # 只返回验证所需的关键列
    print(f"Windows: {window_count}, time gap threshold: {time_gap_threshold:.3f} s, raw_diff step threshold: {used_step_threshold:.6f} m")
    df_res = df_merged[['gps_time', 'window_id', 'window_bias_m', 'kbr_true_range',
                        'gnss_range', 'raw_diff', 'residual', 'residual_cm']].copy()
    
    return df_res, stats

def plot_kbr_residuals(df_res):
    """
    绘制 KBR 残差的时序图与分布直方图 (单位: mm)，适用于高精度定轨验证。
    
    参数:
    - df_res: 包含 'gps_time' 和 'residual' (单位: 米) 列的 DataFrame
    """
    print("正在生成残差可视化图表 (单位: mm)...")
    
    # 提取数据：将 GPS 时间转换为自本弧段起始时刻的流逝时间 (小时)
    t_start = df_res['gps_time'].iloc[0]
    elapsed_hours = (df_res['gps_time'] - t_start) / 3600.0
    
    # 【核心修改】：直接读取原始米级残差并转换为毫米 (mm)
    residuals_mm = df_res['residual'].values * 1000.0
    
    # 计算统计指标 (单位：mm)
    rms = np.sqrt(np.mean(residuals_mm**2))
    mean_val = np.mean(residuals_mm)
    std_val = np.std(residuals_mm)
    
    # 设置学术排版风格
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.autolayout': True
    })

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    # ================= 1. 上图：残差时序图 =================
    ax1.plot(elapsed_hours, residuals_mm, color='#1f77b4', marker='.', linestyle='-', linewidth=1, markersize=3, alpha=0.8, label='Residuals')
    ax1.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # 添加 1倍 STD 的参考线
    ax1.axhline(std_val, color='red', linestyle=':', linewidth=1.5, alpha=0.6, label=r'$\pm 1 \sigma$')
    ax1.axhline(-std_val, color='red', linestyle=':', linewidth=1.5, alpha=0.6)
    
    ax1.set_title('GNSS vs KBR Inter-satellite Range Residuals')
    ax1.set_xlabel('Time since arc start (Hours)')
    ax1.set_ylabel('Residual (mm)')  # 单位更新
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 统计信息框 (单位更新)
    stats_text = f"Points: {len(residuals_mm)}\nMean: {mean_val:.2f} mm\nSTD: {std_val:.2f} mm\nRMS: {rms:.2f} mm"
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    ax1.text(0.98, 0.95, stats_text, transform=ax1.transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='right', bbox=props)
    
    ax1.legend(loc='upper left')

    # ================= 2. 下图：残差分布直方图 =================
    bins = np.linspace(min(residuals_mm), max(residuals_mm), 50)
    ax2.hist(residuals_mm, bins=bins, density=True, color='#aec7e8', edgecolor='black', alpha=0.7)
    
    # 正态分布拟合
    from scipy.stats import norm
    xmin, xmax = ax2.get_xlim()
    x_pdf = np.linspace(xmin, xmax, 100)
    y_pdf = norm.pdf(x_pdf, mean_val, std_val)
    ax2.plot(x_pdf, y_pdf, 'r--', linewidth=2, label='Normal Distribution Fit')
    
    ax2.axvline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_title('Residuals Distribution')
    ax2.set_xlabel('Residual (mm)')  # 单位更新
    ax2.set_ylabel('Density')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right')
    
    # 保存并显示
    plt.tight_layout()
    plt.savefig('KBR_Residuals_Analysis_mm.png', dpi=300, bbox_inches='tight')
    plt.show()

def kbr_analysis(kbr_path, sp3_A_path, sp3_B_path, starttime=None, endtime=None, droprate=None,
                 time_gap_threshold=30.0, residual_step_threshold=None,
                 residual_step_sigma=8.0, min_step_threshold=0.02):
    kbr_df = read_kbr(kbr_path)
    kbr_clean_df = process_kbr(kbr_df)

    sp3_data_A = Sp3Tools.SP3(sp3_A_path)
    sp3_data_B = Sp3Tools.SP3(sp3_B_path)

    df_res, stats = calculate_kbr_residuals(
        sp3_data_A.asDataFrame(),
        sp3_data_B.asDataFrame(),
        kbr_clean_df,
        starttime,
        endtime,
        time_gap_threshold=time_gap_threshold,
        residual_step_threshold=residual_step_threshold,
        residual_step_sigma=residual_step_sigma,
        min_step_threshold=min_step_threshold
    )

    if df_res is None:
        return df_res, stats

    if droprate is not None and 0 < droprate < 1:
        drop_index = get_outlier_indices(df_res['residual'], droprate)
        df_res = df_res.drop(df_res.index[drop_index]).reset_index(drop=True)

    if df_res is not None:
        plot_kbr_residuals(df_res)
    
    return df_res, stats

def convert2rtn(vec, r, v):
    r_norm = np.linalg.norm(r)
    r_unit = r / r_norm
    v_unit = v / np.linalg.norm(v)
    c_unit = np.cross(r_unit, v_unit)
    
    R = np.vstack((r_unit, c_unit, np.cross(c_unit, r_unit)))
    
    return R @ vec

def rmse(a, b):
    return np.sqrt(np.mean((a - b)**2))

def orbit_cmp(sp3_A_path, sp3_B_path, starttime=None, endtime=None, droprate=0.01):
    sp3_data_A = Sp3Tools.SP3(sp3_A_path)
    sp3_data_B = Sp3Tools.SP3(sp3_B_path)
    df_A = sp3_data_A.asDataFrame()
    df_B = sp3_data_B.asDataFrame()
    if not sp3_data_B.withVelocity:
        print("SP3 B does not contain velocity data. Velocity comparison will be skipped.")
        df_B['vx'] = np.zeros(len(df_B))
        df_B['vy'] = np.zeros(len(df_B))
        df_B['vz'] = np.zeros(len(df_B))

    df_cmp = pd.merge(df_A, df_B, on='timesystem', suffixes=('_A', '_B'))
    if starttime is not None:
        df_cmp = df_cmp[df_cmp['timesystem'] >= starttime]
    if endtime is not None:
        df_cmp = df_cmp[df_cmp['timesystem'] <= endtime]
    df_cmp = df_cmp.sort_values('timesystem').reset_index(drop=True)

    df_cmp['dx'] = (df_cmp['x_B'] - df_cmp['x_A'])*1000000.0
    df_cmp['dy'] = (df_cmp['y_B'] - df_cmp['y_A'])*1000000.0
    df_cmp['dz'] = (df_cmp['z_B'] - df_cmp['z_A'])*1000000.0
    df_cmp['dvx'] = (df_cmp['vx_B'] - df_cmp['vx_A'])*10000.0
    df_cmp['dvy'] = (df_cmp['vy_B'] - df_cmp['vy_A'])*10000.0
    df_cmp['dvz'] = (df_cmp['vz_B'] - df_cmp['vz_A'])*10000.0

    time_list = []
    dpos_list = []
    dvel_list = []
    dpos_rtn_list = []
    dvel_rtn_list = []
    for idx, row in df_cmp.iterrows():
        time = row['timesystem']
        time_list.append(time.datetime)
        r = np.array([row['x_A'], row['y_A'], row['z_A']])
        v = np.array([row['vx_A'], row['vy_A'], row['vz_A']])
        dpos = np.array([row['dx'], row['dy'], row['dz']])
        dvel = np.array([row['dvx'], row['dvy'], row['dvz']])
        dpos_rtn = convert2rtn(dpos, r, v)
        dvel_rtn = convert2rtn(dvel, r, v)
        dpos_list.append(dpos)
        dvel_list.append(dvel)
        dpos_rtn_list.append(dpos_rtn)
        dvel_rtn_list.append(dvel_rtn)
        
    df_cmp['time'] = time_list
    df_cmp['dx'] = [d[0] for d in dpos_list]
    df_cmp['dy'] = [d[1] for d in dpos_list]
    df_cmp['dz'] = [d[2] for d in dpos_list]
    df_cmp['dvx'] = [d[0] for d in dvel_list]
    df_cmp['dvy'] = [d[1] for d in dvel_list]
    df_cmp['dvz'] = [d[2] for d in dvel_list]
    df_cmp['dr'] = [d[0] for d in dpos_rtn_list]
    df_cmp['dt'] = [d[1] for d in dpos_rtn_list]
    df_cmp['dn'] = [d[2] for d in dpos_rtn_list]
    df_cmp['dv_r'] = [d[0] for d in dvel_rtn_list]
    df_cmp['dv_t'] = [d[1] for d in dvel_rtn_list]
    df_cmp['dv_n'] = [d[2] for d in dvel_rtn_list]
    df_cmp['dpos_3d'] = np.sqrt(df_cmp['dx']**2 + df_cmp['dy']**2 + df_cmp['dz']**2)
    df_cmp['dvel_3d'] = np.sqrt(df_cmp['dvx']**2 + df_cmp['dvy']**2 + df_cmp['dvz']**2)

    #找出异常点
    dpos3d = np.abs(np.asarray(df_cmp['dpos_3d']))
    n_drop = int(len(dpos3d) * droprate)
    print(f"Total points: {len(dpos3d)}, dropping top {n_drop} ({droprate*100:.1f}%) outliers based on 3D position difference.")
    drop_indices = np.argsort(dpos3d)[-n_drop:]
    print(drop_indices)
    print(df_cmp.iloc[drop_indices][['timesystem', 'dpos_3d']])
    df_cmp = df_cmp.drop(index=drop_indices).reset_index(drop=True)
    
    df_cmp.to_csv('orbit_comparison.csv', index=False)


    # 2*2的图，分别是位置和速度的比较，在原始坐标系下和rtn坐标系下
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].plot(df_cmp['time'], df_cmp['dx'], label='dx')
    axes[0, 0].plot(df_cmp['time'], df_cmp['dy'], label='dy')
    axes[0, 0].plot(df_cmp['time'], df_cmp['dz'], label='dz')
    axes[0, 0].set_title('Position Difference (mm)')
    dx_rmse = rmse(df_cmp['dx'], np.zeros(len(df_cmp)))
    dy_rmse = rmse(df_cmp['dy'], np.zeros(len(df_cmp)))
    dz_rmse = rmse(df_cmp['dz'], np.zeros(len(df_cmp)))
    print(f"Position RMSE: dx={dx_rmse:.3f} mm, dy={dy_rmse:.3f} mm, dz={dz_rmse:.3f} mm, 3D={np.sqrt(dx_rmse**2 + dy_rmse**2 + dz_rmse**2):.3f} mm")
    axes[0,0].text(0.98, 0.95, f"RMSE: dx={dx_rmse:.3f} mm\n dy={dy_rmse:.3f} mm\n dz={dz_rmse:.3f} mm\n 3D={np.sqrt(dx_rmse**2 + dy_rmse**2 + dz_rmse**2):.3f} mm", transform=axes[0, 0].transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    axes[0, 0].legend()

    axes[0, 1].plot(df_cmp['time'], df_cmp['dvx'], label='dvx')
    axes[0, 1].plot(df_cmp['time'], df_cmp['dvy'], label='dvy')
    axes[0, 1].plot(df_cmp['time'], df_cmp['dvz'], label='dvz')
    axes[0, 1].set_title('Velocity Difference (mm/s)')
    dvx_rmse = rmse(df_cmp['dvx'], np.zeros(len(df_cmp)))
    dvy_rmse = rmse(df_cmp['dvy'], np.zeros(len(df_cmp)))
    dvz_rmse = rmse(df_cmp['dvz'], np.zeros(len(df_cmp)))
    print(f"Velocity RMSE: dvx={dvx_rmse:.3f} mm/s, dvy={dvy_rmse:.3f} mm/s, dvz={dvz_rmse:.3f} mm/s, 3D={np.sqrt(dvx_rmse**2 + dvy_rmse**2 + dvz_rmse**2):.3f} mm/s")
    axes[0, 1].text(0.98, 0.95, f"RMSE: dvx={dvx_rmse:.3f} mm/s\n dvy={dvy_rmse:.3f} mm/s\n dvz={dvz_rmse:.3f} mm/s\n 3D={np.sqrt(dvx_rmse**2 + dvy_rmse**2 + dvz_rmse**2):.3f} mm/s", transform=axes[0, 1].transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    axes[0, 1].legend()

    axes[1, 0].plot(df_cmp['time'], df_cmp['dr'], label='dr')
    axes[1, 0].plot(df_cmp['time'], df_cmp['dt'], label='dt')
    axes[1, 0].plot(df_cmp['time'], df_cmp['dn'], label='dn')
    axes[1, 0].set_title('Position Difference in RTN (mm)')
    dr_rmse = rmse(df_cmp['dr'], np.zeros(len(df_cmp)))
    dt_rmse = rmse(df_cmp['dt'], np.zeros(len(df_cmp)))
    dn_rmse = rmse(df_cmp['dn'], np.zeros(len(df_cmp)))
    print(f"Position RMSE in RTN: dr={dr_rmse:.3f} mm, dt={dt_rmse:.3f} mm, dn={dn_rmse:.3f} mm, 3D={np.sqrt(dr_rmse**2 + dt_rmse**2 + dn_rmse**2):.3f} mm")
    axes[1, 0].text(0.98, 0.95, f"RMSE: dr={dr_rmse:.3f} mm\n dt={dt_rmse:.3f} mm\n dn={dn_rmse:.3f} mm\n 3D={np.sqrt(dr_rmse**2 + dt_rmse**2 + dn_rmse**2):.3f} mm", transform=axes[1, 0].transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    axes[1, 0].legend()

    axes[1, 1].plot(df_cmp['time'], df_cmp['dv_r'], label='dv_r')
    axes[1, 1].plot(df_cmp['time'], df_cmp['dv_t'], label='dv_t')
    axes[1, 1].plot(df_cmp['time'], df_cmp['dv_n'], label='dv_n')
    axes[1, 1].set_title('Velocity Difference in RTN (mm/s)')
    dv_r_rmse = rmse(df_cmp['dv_r'], np.zeros(len(df_cmp)))
    dv_t_rmse = rmse(df_cmp['dv_t'], np.zeros(len(df_cmp)))
    dv_n_rmse = rmse(df_cmp['dv_n'], np.zeros(len(df_cmp)))
    print(f"Velocity RMSE in RTN: dv_r={dv_r_rmse:.3f} mm/s, dv_t={dv_t_rmse:.3f} mm/s, dv_n={dv_n_rmse:.3f} mm/s, 3D={np.sqrt(dv_r_rmse**2 + dv_t_rmse**2 + dv_n_rmse**2):.3f} mm/s")
    axes[1, 1].text(0.98, 0.95, f"RMSE: dv_r={dv_r_rmse:.3f} mm/s\n dv_t={dv_t_rmse:.3f} mm/s\n dv_n={dv_n_rmse:.3f} mm/s\n 3D={np.sqrt(dv_r_rmse**2 + dv_t_rmse**2 + dv_n_rmse**2):.3f} mm/s", transform=axes[1, 1].transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

    
    

if __name__ == '__main__':
    starttime = None 
    endtime = None
    # starttime = TimeSystem.TimeSystem().from_str('2020-09-8 2:00:00.000')
    # endtime = TimeSystem.TimeSystem().from_str('2020-09-13 17:50:00.000')

    # obs_path_edit_A = r"D:\csu\GNSS_MIX\CSUAPPS\CSUPODSApp\BIN\inputdata\Obs\Diff\GFO_C_DataEditing_Obs_code_test.rnx"
    # obs_path_edit_B = r"D:\csu\GNSS_MIX\CSUAPPS\CSUPODSApp\BIN\inputdata\Obs\Diff\GFO_D_DataEditing_Obs_code_test.rnx"
    # obs_path_ori_A = r"D:\csu\GNSS_MIX\CSUAPPS\CSUPODSApp\BIN\inputdata\Obs\Diff\GFCN2450_edited.30O"
    # obs_path_ori_B = r"D:\csu\GNSS_MIX\CSUAPPS\CSUPODSApp\BIN\inputdata\Obs\Diff\GFDN2450_edited.30O"
    # plot_sat_num_diff(obs_path_edit_A, obs_path_edit_B, tag='_edited')
    # plot_sat_num_diff(obs_path_ori_A, obs_path_ori_B, tag='_original')

    
    kbr_path = r"D:\csu\GNSS_MIX\CSUAPPS\CSUPODSApp\BIN\inputdata\GFO_month\KBR\GFO20202550.KBR"
    sp3_A_path = r"D:\csu\GNSS_MIX\CSUAPPS\CSUPODSApp\BIN\outputdata\Diff_batch\GFOD_smooth_20200911.sp3"
    sp3_B_path = r"D:\csu\GNSS_MIX\CSUAPPS\CSUPODSApp\BIN\outputdata\Diff_batch\GFOC_smooth_20200911.sp3"
    # sp3_A_path = r"D:\csu\GNSS_MIX\CSUAPPS\CSUPODSApp\BIN\inputdata\GFO_month\DiffRef\JPL_C_20200911.sp3"
    # sp3_B_path = r"D:\csu\GNSS_MIX\CSUAPPS\CSUPODSApp\BIN\inputdata\GFO_month\DiffRef\JPL_D_20200911.sp3"
    df, _ = kbr_analysis(kbr_path, sp3_A_path, sp3_B_path, starttime, endtime, droprate=None)
    df.to_csv('kbr_residuals.csv', index=False)

    # sp3_A_path = r"D:\csu\GNSS_MIX\CSUAPPS\CSUPODSApp\BIN\outputdata\Diff_GFO_20200901\GFOD_smooth_20200901_rts.sp3"
    # sp3_B_path = r"D:\csu\GNSS_MIX\CSUAPPS\CSUPODSApp\BIN\inputdata\Obs\Diff\JPLD202450.PRE"
    # orbit_cmp(sp3_A_path, sp3_B_path, starttime, endtime)


