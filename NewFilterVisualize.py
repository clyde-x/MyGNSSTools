import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates

import TimeSystem

# ==========================================
# 1. 全局常量定义 (Global Constants)
# ==========================================
CONST_C = 299792458.0          # 光速，单位：m/s
CONST_C_ns = CONST_C * 1e-9    # 光速，单位：m/ns

# ==========================================
# 2. 数据结构定义 (Data Structures)
# ==========================================
@dataclass
class EpochData:
    """定义每个历元(Epoch)的数据结构，替代原本庞大繁杂的字典"""
    timestamp: Any  # TimeSystem.TimeSystem 实例
    
    # 向量状态 (Vectors)
    sc_a_state: Optional[np.ndarray] = None
    sc_b_state: Optional[np.ndarray] = None
    state_vector: Optional[np.ndarray] = None
    sc_b_gt_state: Optional[np.ndarray] = None
    sc_b_state_err: Optional[np.ndarray] = None
    pred_state_vector: Optional[np.ndarray] = None
    pred_residuals: Optional[np.ndarray] = None
    updated_state_vector: Optional[np.ndarray] = None
    post_fit_residuals: Optional[np.ndarray] = None
    smoothed_state_vector: Optional[np.ndarray] = None
    smoothed_sc_b_state_err: Optional[np.ndarray] = None
    cs_sats: Optional[np.ndarray] = None  # 发生周跳的卫星 (Satellites with Cycle Slips)
    smoothed_post_fit_residuals: Optional[np.ndarray] = None
        
    
    # 矩阵状态 (Matrices)
    cov_matrix: Optional[np.ndarray] = None
    state_trans_matrix: Optional[np.ndarray] = None
    process_noise_matrix: Optional[np.ndarray] = None
    pred_cov_matrix: Optional[np.ndarray] = None
    design_matrix_h: Optional[np.ndarray] = None
    kalman_gain_k: Optional[np.ndarray] = None
    updated_cov_matrix: Optional[np.ndarray] = None
    
    # 其他属性 (Others)
    common_sats: List[str] = field(default_factory=list)
    lambda_info: Dict[str, Any] = field(default_factory=dict)


# ==========================================
# 3. 基础工具与解析函数 (Utils & Parsers)
# ==========================================
def extract_array(block, label, shape=None, dtype=float):
    """从文本块中通过正则提取数组或矩阵"""
    pattern = rf"{re.escape(label)},\s*size:\s+(\d+)(?:x(\d+))?"
    m = re.search(pattern, block)
    if not m:
        return None

    if shape == 'matrix':
        rows = int(m.group(1))
        cols = int(m.group(2)) if m.group(2) else rows
        expected_size = rows * cols
    else:  # vector
        expected_size = int(m.group(1))
        rows, cols = expected_size, 1

    start_pos = m.end()
    tail = block[start_pos:]

    if shape == 'SatSet':
        tokens = tail[1:].split('\t')
        tokens = [token.strip() for token in tokens if token.strip() != '']
        if len(tokens) < expected_size:
            print(f"Warning: Not enough tokens for '{label}' (need {expected_size}, got {len(tokens)})")
            return None
        return np.array(tokens[:expected_size], dtype=str).tolist()
    else:
        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", tail)
        if len(numbers) < expected_size:
            print(f"Warning: Not enough numbers for '{label}' (need {expected_size}, got {len(numbers)})")
            return None

        arr = np.array(numbers[:expected_size], dtype=dtype)
        return arr.reshape((rows, cols)) if shape == 'matrix' else arr

def extract_lambda_info(block, last_lambda):
    """提取 LAMBDA 模糊度固定信息"""
    lambda_dict = {'fixed': True}
    
    match_fixed = re.search(r'LAMBDA fix \(L1\+L2\):\s*(OK|NO)\b', block)
    if match_fixed and match_fixed.group(1) == 'NO':
        lambda_dict['fixed'] = False 
    
    pattern = r'prob=(?P<prob>[-+]?\d+\.\d+e[-+]\d+), sqn_s=(?P<sqn_s>[-+]?\d+\.\d+e[-+]\d+), sqn_m=(?P<sqn_m>[-+]?\d+\.\d+e[-+]\d+)'
    match = re.search(pattern, block)
    
    if match:
        lambda_dict['prob'] = float(match.group('prob'))
        lambda_dict['sqn_s'] = float(match.group('sqn_s'))
        lambda_dict['sqn_m'] = float(match.group('sqn_m'))
    else:
        lambda_dict['prob'] = last_lambda.get('prob', np.nan)
        lambda_dict['sqn_s'] = last_lambda.get('sqn_s', np.nan)
        lambda_dict['sqn_m'] = last_lambda.get('sqn_m', np.nan)
    
    return lambda_dict

def RMSE(residuals):
    """计算均方根误差"""
    if residuals is None or residuals.size == 0:
        return np.array([])
    return np.sqrt(np.mean(residuals ** 2, axis=0))

def j2000_to_rtn(r_j2000, v_j2000, error_j2000):
    """将 J2000 惯性系下的误差向量转换到 RTN 轨道坐标系"""
    r, v, err = np.array(r_j2000, dtype=float), np.array(v_j2000, dtype=float), np.array(error_j2000, dtype=float)
    
    r_norm = np.linalg.norm(r)
    if r_norm == 0: raise ValueError("位置向量模长为0，无法定义 RTN 坐标系")
    u_r = r / r_norm
    
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    if h_norm == 0: raise ValueError("角动量为0 (直线运动)，无法定义 RTN 坐标系")
    u_n = h / h_norm
    
    u_t = np.cross(u_n, u_r)
    rot_matrix = np.array([u_r, u_t, u_n])
    # rot_matrix = np.vstack((u_r, u_t, u_n)).T  # 3x3 转置为列向量形式
    
    return rot_matrix @ err

# ==========================================
# 4. 核心生成器与文件处理 (Generators)
# ==========================================
def split_file_by_marker(input_file):
    base_name, ext = os.path.splitext(input_file)
    files_map = {
        'forward': f"{base_name}_forward{ext}",
        'backward': f"{base_name}_backward{ext}",
        'twoway': f"{base_name}_twoway{ext}",
        'kinematics': f"{base_name}_kinematics{ext}"
    }
    
    markers = {
        "*** backward filtering ***": 'backward',
        "Start Smoothing.": 'twoway',
        "Start Kinematics.": 'kinematics'
    }
    
    current_mode = 'forward'
    current_file_handle = open(files_map[current_mode], 'w', encoding='utf-8')
    generated_files = [files_map[current_mode]]
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            for marker, mode in markers.items():
                if marker in line:
                    current_file_handle.close()
                    current_mode = mode
                    current_file_handle = open(files_map[current_mode], 'w', encoding='utf-8')
                    generated_files.append(files_map[current_mode])
                    break
            current_file_handle.write(line)
            
    current_file_handle.close()
    print(f"已生成：{' 和 '.join(generated_files)}")
    return generated_files


def generate_epoch_blocks(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        block = []
        for line in f:
            if "Current Epoch:" in line:
                if block:
                    yield "".join(block)
                block = [line]
            elif block:
                block.append(line)
        if block:
            yield "".join(block)


def parse_kalman_log_with_header(filepath, starttime=None, endtime=None) -> List[EpochData]:
    epochs_data = []
    last_lambda = {'fixed': False}

    # 使用映射字典将日志字符串对应到 Dataclass 的属性上
    vector_keys_map = {
        "Spacecraft A State": "sc_a_state",
        "Spacecraft B State": "sc_b_state",
        "State Vector": "state_vector",
        "Spacecraft B Ground Truth State": "sc_b_gt_state",
        "Spacecraft B State Error (GT - Est)": "sc_b_state_err",
        "Predicted State Vector": "pred_state_vector",
        "Predicted Residuals": "pred_residuals",
        "Updated State Vector": "updated_state_vector",
        "Post-fit Residuals": "post_fit_residuals",
        "Smoothed State Vector": "smoothed_state_vector",
        "Smoothed Spacecraft B State Error (GT - Est)": "smoothed_sc_b_state_err",
        "Satellites with Cycle Slips": "cs_sats",
        "Smoothed Post-Fit Residuals": "smoothed_post_fit_residuals",
        "RTS Smoothed Spacecraft B State Error (GT - Est)": "rts_error"
    }

    matrix_keys_map = {
        "Covariance Matrix": "cov_matrix",
        "State Transition Matrix": "state_trans_matrix",
        "Process Noise Matrix": "process_noise_matrix",
        "Predicted Covariance Matrix": "pred_cov_matrix",
        "Design Matrix H": "design_matrix_h",
        "Kalman Gain K": "kalman_gain_k",
        "Updated Covariance Matrix": "updated_cov_matrix"
    }

    for block in generate_epoch_blocks(filepath):
        block_content = block.replace("Current Epoch:", "", 1).strip()
        timestamp_match = re.match(r'^\s*([\d:\.]+)', block_content)
        
        if not timestamp_match:
            continue
            
        time = TimeSystem.TimeSystem().from_str('2020-09-01 ' + timestamp_match.group(1))
        if (starttime and time < starttime) or (endtime and time > endtime):
            continue

        # 初始化 Dataclass 对象
        epoch_obj = EpochData(timestamp=time)

        # 反射注入向量数据
        for log_key, attr_name in vector_keys_map.items():
            setattr(epoch_obj, attr_name, extract_array(block, log_key, shape='vector'))
            
        # 反射注入矩阵数据
        for log_key, attr_name in matrix_keys_map.items():
            setattr(epoch_obj, attr_name, extract_array(block, log_key, shape='matrix'))
            
        epoch_obj.common_sats = extract_array(block, "Common Satellites", shape='SatSet') or []
        last_lambda = extract_lambda_info(block, last_lambda)
        epoch_obj.lambda_info = last_lambda
        
        epochs_data.append(epoch_obj)

    return epochs_data

# ==========================================
# 5. 绘图相关函数 (Visualization)
# ==========================================
def plot_state(time_list, data_list, name_list, title, ylabel, scatter=False, all_axis=False, savefig_path=None, print_rmse=False, print_mean_std=False, print_std=False, cs_time_list=[], savecsv=False):
    plt.figure(figsize=(10, 6))
    if data_list.shape[1] != len(name_list):
        raise ValueError("Data list length and name list length do not match.")
        
    for i in range(data_list.shape[1]):
        # ==========================================
        # 新增核心逻辑：判断是否在图例中加入 std
        # ==========================================
        if print_std:
            current_std = np.std(data_list[:, i])
            label_text = f"{name_list[i]} (std={current_std:.4f})"
        else:
            label_text = name_list[i]
            
        if scatter:
            plt.scatter(time_list, data_list[:, i], label=label_text)
        else:
            plt.plot(time_list, data_list[:, i], label=label_text)
            
    # 获取当前的坐标轴对象，方便后续统一设置
    ax = plt.gca()
            
    if print_rmse:
        rmses = [RMSE(data_list[:, i]) for i in range(data_list.shape[1])] # 注意：RMSE 函数需要在外部有定义
        print(f"RMSE of {name_list}: {rmses}")
        
        # 【放大字体】(20 -> 24) 并添加白底背景避免和数据线条重叠
        plt.text(0.5, 0.01, 'RMSE: ' + ', '.join([f"{name_list[i]}: {rmses[i]:.4f}" for i in range(len(rmses))]), 
                 ha='center', va='bottom', transform=ax.transAxes, fontsize=24, 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))
                 
        if all_axis:
            rmses = np.array(rmses)
            rmses_all = np.sqrt(np.sum(rmses**2))
            print(f"Overall RMSE of all axes: {rmses_all:.4f}")
            
            # 【放大字体】(20 -> 24) 并将 y=0.05 微调为 0.09，避免大号字体互相重叠
            plt.text(0.5, 0.09, f'Overall RMSE: {rmses_all:.4f}', ha='center', va='bottom', 
                     transform=ax.transAxes, fontsize=24, color='red', 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))
        
    if print_mean_std:
        means = [np.mean(data_list[:, i]) for i in range(data_list.shape[1])]
        stds = [np.std(data_list[:, i]) for i in range(data_list.shape[1])]
        
        # 【放大字体】(20 -> 24) 添加白底背景
        plt.text(0.5, 0.01, 'Mean & Std: ' + ', '.join([f"{name_list[i]}: {means[i]:.4f} ± {stds[i]:.4f}" for i in range(len(means))]), 
                 ha='center', va='bottom', transform=ax.transAxes, fontsize=24, 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))
    
    for cs_time in cs_time_list:
        plt.axvline(x=cs_time.datetime, color='r', linestyle='--', alpha=0.5)
        
    # 【放大字体】主标题和坐标轴标签
    plt.title(title, fontsize=26, fontweight='bold')
    plt.xlabel("Time", fontsize=22)
    plt.ylabel(ylabel, fontsize=22)
    
    # 【放大字体】坐标轴刻度数字
    ax.tick_params(axis='both', labelsize=18)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    
    # 【放大字体】图例
    plt.legend(fontsize=18)
    
    if savefig_path:
        plt.savefig(f"{savefig_path}/{title.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    if savecsv:
        df = pd.DataFrame(data_list, columns=name_list)
        df.insert(0, 'Time', time_list)
        df.to_csv(f"{savefig_path}/{title.replace(' ', '_')}.csv", index=False)


import matplotlib.dates as mdates

def plot_lambda_fix_analysis(time_arr, fixed_arr, title="LAMBDA Fix Analysis", window_size=60, savefig_path=None):
    """
    专门针对 0/1 状态的 LAMBDA Fix 标志进行高级可视化
    包含上下两个子图：
    上图：滑动窗口固定率 (展示趋势)
    下图：事件图/条形码图 (展示离散的固定丢失情况)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    
    # ------------------ 子图 1: 滑动窗口固定率 ------------------
    # 使用 Pandas 计算滚动平均值（固定率）
    fixed_series = pd.Series(fixed_arr)
    rolling_rate = fixed_series.rolling(window=window_size, min_periods=1).mean() * 100
    
    ax1.plot(time_arr, rolling_rate, color='blue', linewidth=1.5)
    ax1.fill_between(time_arr, rolling_rate, 100, color='red', alpha=0.2) # 将未达到 100% 的区域标红预警
    
    # 【放大字体】Y轴标签和主标题
    ax1.set_ylabel(f'Fix Rate (%) \n({window_size}-epoch window)', fontsize=22)
    ax1.set_title(title, fontsize=26, fontweight='bold') 
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 【放大字体】坐标轴上的刻度数字 (Y轴)
    ax1.tick_params(axis='y', labelsize=18)

    # ------------------ 子图 2: 条形码/事件图 ------------------
    # 提取 fixed=1 和 fixed=0 的时间点
    time_arr_np = np.array(time_arr)
    fixed_times = time_arr_np[np.array(fixed_arr) == 1]
    unfixed_times = time_arr_np[np.array(fixed_arr) == 0]

    # 画竖线 (ymin=0, ymax=1)
    if len(unfixed_times) > 0:
        ax2.vlines(unfixed_times, ymin=0, ymax=1, colors='red', alpha=0.8, linewidth=0.5, label='Unfixed (0)')
    if len(fixed_times) > 0:
        ax2.vlines(fixed_times, ymin=0, ymax=1, colors='green', alpha=0.3, linewidth=0.5, label='Fixed (1)')

    # 【放大字体】Y轴标签
    ax2.set_ylabel('Status', fontsize=22)
    ax2.set_yticks([]) # 隐藏 Y 轴刻度
    
    # 【放大字体】图例
    ax2.legend(loc='upper right', fontsize=18)
    
    # 设置 X 轴时间格式
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # 【放大字体】X轴总体标签
    plt.xlabel("Time", fontsize=22)
    
    # 【放大字体】坐标轴上的时间刻度数字 (X轴)
    ax2.tick_params(axis='x', labelsize=18)
    
    # ------------------ 整体状态统计文字 ------------------
    total = len(fixed_arr)
    fixed_cnt = np.sum(fixed_arr)
    stats_text = f"Total Epochs: {total} | Fixed: {fixed_cnt} | Unfixed: {total - fixed_cnt} | Overall Rate: {(fixed_cnt/total)*100:.2f}%"
    
    # 【放大字体】底部统计文本
    plt.figtext(0.5, -0.05, stats_text, ha="center", fontsize=24, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    plt.tight_layout()
    
    if savefig_path:
        plt.savefig(f"{savefig_path}/{title.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

def plot_iono(time_arr, iono_arr, savefig_path=None):
    """
    可视化电离层延时 (横轴：时间，纵轴：电离层延时)
    
    参数:
    time_arr : list or array of datetime objects
    iono_arr : list of arrays/lists, 每个元素是当前历元所有可见卫星的电离层延时
    savefig_path : 保存图片的路径目录
    """
    plot_x = []
    plot_y = []

    # 1. 数据扁平化配对
    for t, ionos in zip(time_arr, iono_arr):
        if ionos is None or len(ionos) == 0:
            continue
            
        for val in ionos:
            # 建议过滤掉无效值 (如 NaN)，防止绘图报错
            if not np.isnan(val):
                plot_x.append(t)
                plot_y.append(val)

    # 2. 开始绘图
    plt.figure(figsize=(12, 6))
    
    # 使用散点图，因为数据点可能非常密集，建议调小 size (s) 并增加透明度 (alpha)
    plt.scatter(plot_x, plot_y, s=5, alpha=0.5, c='royalblue', marker='.')

    # 3. 设置图表属性及【放大字体】
    plt.title("Ionospheric Delay Over Time", fontsize=26, fontweight='bold') # 放大主标题并加粗
    plt.xlabel("Time", fontsize=22)                                        # 放大 X 轴标签
    plt.ylabel("Ionospheric Delay (m)", fontsize=22)                       # 放大 Y 轴标签
    
    # 自动格式化 X 轴的时间显示 (例如显示为 小时:分钟)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # 【全新加入：放大坐标轴刻度数字】
    ax.tick_params(axis='both', labelsize=18) # 统一放大 X 轴和 Y 轴的刻度数字
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # 4. 保存或显示
    if savefig_path:
        # 确保保存路径存在
        os.makedirs(savefig_path, exist_ok=True)
        plt.savefig(f"{savefig_path}/Ionospheric_Delay.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
        
    plt.close()

def plot_clk_bias(time_arr, clk_arr, drop_rate = 0.03, window_size=21, savefig_path=None, print_mean_std=False, cs_time_list=[]):
    if drop_rate <= 0.0 or drop_rate >= 1.0:
        return time_arr, clk_arr
    if len(time_arr) != len(clk_arr):
        raise ValueError("time_arr 和 clk_arr 长度必须一致！")
    
    s_clk = pd.Series(clk_arr)
    rolling_median = s_clk.rolling(window=window_size, center=True, min_periods=1).median() 
    deviations = np.abs(s_clk - rolling_median)
    percentile_threshold = (1.0 - drop_rate) * 100
    cutoff_value = np.percentile(deviations.dropna(), percentile_threshold)

    valid_mask = deviations <= cutoff_value
    filtered_time = time_arr[valid_mask]
    filtered_clk = clk_arr[valid_mask]

    original_len = len(clk_arr)
    filtered_len = len(filtered_clk)
    print(f"Hampel (droprate={drop_rate}) -> 原长度: {original_len}, 剔除: {original_len - filtered_len} 点, 剩余: {filtered_len}")

    plot_state(filtered_time, filtered_clk.reshape(-1, 1), ['Clock Bias'], f"Clock Bias(drop_rate={drop_rate})", "Clock Bias (ns)", savefig_path=savefig_path, print_mean_std=print_mean_std, cs_time_list=cs_time_list)

    df = pd.DataFrame({'Time': filtered_time, 'Clock Bias (ns)': filtered_clk})
    df.to_csv(f"{savefig_path}/Clock_Bias_filtered.csv", index=False)

# ==========================================
# 6. 状态解析与核心业务逻辑 (State Parsing)
# ==========================================
def parse_SC_state(epochs_data: List[EpochData], savefig_path=None, Cr=True, Cd=True, Acc=True, KBR = False):
    basic_state = 6
    dyn_state = 0
    cr_index, cd_index, acc_index, kbr_index = None, None, None, None
    
    if Cr:
        dyn_state += 1
        cr_index = basic_state + dyn_state - 1
    if Cd:
        dyn_state += 1
        cd_index = basic_state + dyn_state - 1
    if Acc:
        dyn_state += 3
        acc_index = basic_state + dyn_state - 3
    if KBR:
        dyn_state += 1
        kbr_index = basic_state + dyn_state - 1
    clk_index = basic_state + dyn_state

    history = defaultdict(list)
    cs_time_list = []

    for epoch in epochs_data:
        if len(epoch.state_vector) != clk_index + 1 + len(epoch.common_sats) * 3:
            print(f"Warning: Epoch at {epoch.timestamp} has unexpected state vector length {len(epoch.state_vector)} (expected {clk_index + 1 + epoch.sat_num * 3})")
            continue
        history['time'].append(epoch.timestamp.datetime)
        history['sats'].append(epoch.common_sats)
        history['sat_num'].append(len(epoch.common_sats))
        
        # Spacecraft A
        history['A_pos'].append(epoch.sc_a_state[0:3] / 1000.0)
        history['A_vel'].append(epoch.sc_a_state[3:6])
        history['A_force'].append(epoch.sc_a_state[cr_index:cd_index+1] if Cr and Cd else [np.nan, np.nan])
        history['A_acc'].append(epoch.sc_a_state[acc_index:acc_index+3] * 1e9 if Acc else [np.nan, np.nan, np.nan])

        # Spacecraft B
        history['B_pos'].append(epoch.sc_b_state[0:3] / 1000.0)
        history['B_vel'].append(epoch.sc_b_state[3:6])
        history['B_force'].append(epoch.sc_b_state[cr_index:cd_index+1] if Cr and Cd else [np.nan, np.nan])
        history['B_acc'].append(epoch.sc_b_state[acc_index:acc_index+3] * 1e9 if Acc else [np.nan, np.nan, np.nan])
        
        # Ground Truth & Residuals
        history['B_pos_gt'].append(epoch.sc_b_gt_state[0:3] / 1000.0)
        history['B_pos_res'].append(epoch.sc_b_state_err[0:3] * 100.0)
        history['B_vel_gt'].append(epoch.sc_b_gt_state[3:6])
        history['B_vel_res'].append(epoch.sc_b_state_err[3:6] * 100.0)
        sats_num = int (len(epoch.post_fit_residuals)/4)
        history['post_res'].append(RMSE(epoch.post_fit_residuals[:sats_num*2]) * 100.0) #cm
        history['post_res_code'].append(RMSE(epoch.post_fit_residuals[:sats_num*2]) * 100.0) #cm
        history['post_res_phase'].append(RMSE(epoch.post_fit_residuals[sats_num*2:sats_num*4]) * 100.0) #cm

        # Relative States
        history['rel_pos'].append(epoch.state_vector[0:3] / 1000.0)
        history['rel_vel'].append(epoch.state_vector[3:6])
        history['rel_force'].append(epoch.state_vector[cr_index:cd_index+1] if Cr and Cd else [np.nan, np.nan])
        history['rel_acc'].append(epoch.state_vector[acc_index:acc_index+3] * 1e9 if Acc else [np.nan, np.nan, np.nan])
        history['rel_clk'].append(epoch.state_vector[clk_index] / CONST_C_ns)
        # 没有kbr

        history['iono'].append(epoch.state_vector[clk_index+1:clk_index+1+sats_num])
        history['amb1'].append(epoch.state_vector[clk_index+1+sats_num:clk_index+1+2*sats_num])
        history['amb2'].append(epoch.state_vector[clk_index+1+2*sats_num:clk_index+1+3*sats_num])

        if epoch.cs_sats is not None:
            cs_time_list.append(epoch.timestamp)
        history['cs_sats'].append(epoch.cs_sats)
        history['cs_sats_num'].append(len(epoch.cs_sats) if epoch.cs_sats is not None else 0)
            
            
        history['fixed'].append(1 if epoch.lambda_info.get('fixed') else 0)

    for key in history:
        if key in ['sats', 'iono', 'amb1', 'amb2', 'cs_sats']:
            continue  # 这些字段是列表或字符串，不适合转换为 NumPy 数组
        history[key] = np.array(history[key])

    time_arr = history['time']
    
    # 绘图操作 (部分代码省略冗余展示)
    plot_state(time_arr, history['sat_num'].reshape(-1, 1), ['Sat Count'], "Common Satellite Count", "Count", savefig_path=savefig_path, print_mean_std=True)
    # plot_state(time_arr, history['A_pos'], ['X', 'Y', 'Z'], "Spacecraft A Position", "Position (km)", savefig_path=savefig_path)
    # plot_state(time_arr, history['A_vel'], ['Vx', 'Vy', 'Vz'], "Spacecraft A Velocity", "Velocity (m/s)", savefig_path=savefig_path)
    # if Cr and Cd: plot_state(time_arr, history['A_force'], ['Cd', 'Cr'], "Spacecraft A Force Coefficients", "Cd and Cr", savefig_path=savefig_path)
    # if Acc: plot_state(time_arr, history['A_acc'], ['ar', 'at', 'an'], "Spacecraft A Accelerations", "EmpAcc (nm/s²)", savefig_path=savefig_path)
    
    # plot_state(time_arr, history['B_pos'], ['X', 'Y', 'Z'], "Spacecraft B Position", "Position (km)", savefig_path=savefig_path)
    # plot_state(time_arr, history['B_vel'], ['Vx', 'Vy', 'Vz'], "Spacecraft B Velocity", "Velocity (m/s)", savefig_path=savefig_path)
    # cs_time_list = []
    plot_clk_bias(time_arr, history['rel_clk'], savefig_path=savefig_path, print_mean_std=True)
    # plot_state(time_arr, history['B_pos_gt'], ['X_gt', 'Y_gt', 'Z_gt'], "Spacecraft B Ground Truth Position", "Position (km)", savefig_path=savefig_path)
    plot_state(time_arr, history['B_pos_res'], ['X_res', 'Y_res', 'Z_res'], "Spacecraft B Position Residuals (GT - Est)", "Position Residuals (cm)", savefig_path=savefig_path, print_rmse=True,all_axis=True, cs_time_list=cs_time_list)
    plot_state(time_arr, history['B_vel_res'], ['Vx_res', 'Vy_res', 'Vz_res'], "Spacecraft B Velocity Residuals (GT - Est)", "Velocity Residuals (cm/s)", savefig_path=savefig_path, print_rmse=True, all_axis=True, cs_time_list=cs_time_list)
    plot_state(time_arr, history['post_res_code'].reshape(-1, 1), ['Code Residuals'], "Post-Fit Residuals (Code)", "Post-Fit Residuals (cm)", savefig_path=savefig_path, print_rmse=True, all_axis=True, cs_time_list=cs_time_list)
    plot_state(time_arr, history['post_res_phase'].reshape(-1, 1), ['Phase Residuals'], "Post-Fit Residuals (Phase)", "Post-Fit Residuals (cm)", savefig_path=savefig_path, print_rmse=True, all_axis=True, cs_time_list=cs_time_list)
    
    # plot_state(time_arr, history['rel_pos'], ['X_AB', 'Y_AB', 'Z_AB'], "Relative Position (A-B)", "Relative Position (km)", savefig_path=savefig_path)
    # plot_state(time_arr, history['rel_vel'], ['Vx_AB', 'Vy_AB', 'Vz_AB'], "Relative Velocity (A-B)", "Relative Velocity (m/s)", savefig_path=savefig_path)
    plot_state(time_arr, history['rel_force'], ['Cd', 'Cr'], "Relative Force Coefficients", "Cd and Cr", savefig_path=savefig_path)
    plot_state(time_arr, history['rel_acc'], ['ar', 'at', 'an'], "Relative Accelerations", "EmpAcc (nm/s²)", savefig_path=savefig_path, print_std=True)
    plot_state(time_arr, history['cs_sats_num'].reshape(-1, 1), ['CS Sat Count'], "Cycle Slip Satellite Count", "Count", savefig_path=savefig_path)

    # 修改为新的高级可视化代码：
    plot_lambda_fix_analysis(time_arr, history['fixed'], title="", window_size=60, savefig_path=savefig_path)

    plot_iono(time_arr, history['iono'], savefig_path=savefig_path)

    fixed_count = np.sum(history['fixed'])
    print(f"fixed count: {fixed_count}, total count: {len(history['fixed'])}, fixed rate: {fixed_count/len(history['fixed']):.4f}")

def parse_two_way_state(epochs_data: List[EpochData], savefig_path=None, Cr=True, Cd=True, Acc=True, KBR=False):
    basic_state = 6
    dyn_state = 0
    cd_index, cr_index, acc_index, kbr_index = None, None, None, None
    if Cd:
        dyn_state += 1; cd_index = basic_state + dyn_state - 1
    if Cr:
        dyn_state += 1; cr_index = basic_state + dyn_state - 1
    if Acc:
        dyn_state += 3; acc_index = basic_state + dyn_state - 3
    if KBR:
        dyn_state += 1; kbr_index = basic_state + dyn_state - 1

    clk_index = basic_state + dyn_state

    history = defaultdict(list)
    cs_time_list = []

    for epoch in epochs_data:
        history['time'].append(epoch.timestamp.datetime)
        history['pos'].append(epoch.state_vector[0:3] / 1000.0)
        history['pos_res'].append(epoch.smoothed_sc_b_state_err[0:3] * 100.0)
        history['vel'].append(epoch.state_vector[3:6])
        history['vel_res'].append(epoch.smoothed_sc_b_state_err[3:6] * 1000.0)
        history['force'].append(epoch.state_vector[cd_index:cr_index+1] if Cd and Cr else [np.nan, np.nan])
        history['acc'].append(epoch.state_vector[acc_index:acc_index+3] * 1e9 if Acc else [np.nan, np.nan, np.nan])
        history['clk'].append(epoch.state_vector[clk_index] / CONST_C_ns)
        # usephase 
        sats_num = int (len(epoch.smoothed_post_fit_residuals)/4)
        history['smooth_post_res_code'].append(RMSE(epoch.smoothed_post_fit_residuals[:sats_num*2]) * 100.0) #cm
        history['smooth_post_res_phase'].append(RMSE(epoch.smoothed_post_fit_residuals[sats_num*2:sats_num*4]) * 100.0) #cm 

        history['iono'].append(epoch.state_vector[clk_index+1:clk_index+1+sats_num])
        history['amb1'].append(epoch.state_vector[clk_index+1+sats_num:clk_index+1+2*sats_num])
        history['amb2'].append(epoch.state_vector[clk_index+1+2*sats_num:clk_index+1+3*sats_num])

        if epoch.cs_sats is not None:
            cs_time_list.append(epoch.timestamp)
        history['cs_sats'].append(epoch.cs_sats)
        history['cs_sats_num'].append(len(epoch.cs_sats) if epoch.cs_sats is not None else 0)

        if epoch.rts_error is not None:
            history['rts_pos_res'].append(epoch.rts_error[0:3] * 100.0)
            history['rts_vel_res'].append(epoch.rts_error[3:6] * 1000.0)

    for key in history:
        if key in ['sats', 'iono', 'amb1', 'amb2', 'cs_sats']:
            continue  # 这些字段是列表或字符串，不适合转换为 NumPy 数组
        history[key] = np.array(history[key])

    time_arr = history['time']
    # plot_state(time_arr, history['pos'], ['X_AB', 'Y_AB', 'Z_AB'], "Two-Way Relative Position (A-B)", "Relative Position (km)", savefig_path=savefig_path)
    # plot_state(time_arr, history['vel'], ['Vx_AB', 'Vy_AB', 'Vz_AB'], "Two-Way Relative Velocity (A-B)", "Relative Velocity (m/s)", savefig_path=savefig_path)
    plot_clk_bias(time_arr, history['clk'], savefig_path=savefig_path, print_mean_std=True, cs_time_list=cs_time_list)
    plot_state(time_arr, history['pos_res'], ['X_res', 'Y_res', 'Z_res'], "Two-Way Relative Position Residuals (GT - Est)", "Position Residuals (cm)", savefig_path=savefig_path, print_rmse=True,all_axis=True, cs_time_list=cs_time_list)
    plot_state(time_arr, history['vel_res'], ['Vx_res', 'Vy_res', 'Vz_res'], "Two-Way Relative Velocity Residuals (GT - Est)", "Velocity Residuals (mm/s)", savefig_path=savefig_path, print_rmse=True, all_axis=True, cs_time_list=cs_time_list)
    plot_state(time_arr, history['smooth_post_res_code'].reshape(-1, 1), ['Code Residuals'], "Smoothed Post-Fit Residuals (Code)", "Post-Fit Residuals (cm)", savefig_path=savefig_path, print_rmse=True, all_axis=True, cs_time_list=cs_time_list)
    plot_state(time_arr, history['smooth_post_res_phase'].reshape(-1, 1), ['Phase Residuals'], "Smoothed Post-Fit Residuals (Phase)", "Post-Fit Residuals (cm)", savefig_path=savefig_path, print_rmse=True, all_axis=True, cs_time_list=cs_time_list)
    # plot_state(time_arr, history['cs_sats_num'].reshape(-1, 1), ['CS Sat Count'], "Cycle Slip Satellite Count", "Number of Satellites with Cycle Slips", savefig_path=savefig_path)

    plot_state(time_arr, history['rts_pos_res'], ['X_rts_res', 'Y_rts_res', 'Z_rts_res'], "RTS Smoothed Relative Position Residuals (GT - Est)", "Position Residuals (cm)", savefig_path=savefig_path, print_rmse=True, all_axis=True, cs_time_list=cs_time_list)
    plot_state(time_arr, history['rts_vel_res'], ['Vx_rts_res', 'Vy_rts_res', 'Vz_rts_res'], "RTS Smoothed Relative Velocity Residuals (GT - Est)", "Velocity Residuals (mm/s)", savefig_path=savefig_path, print_rmse=True, all_axis=True, cs_time_list=cs_time_list)
    plot_iono(time_arr, history['iono'], savefig_path=savefig_path)

def parse_kin_state(filepath, savefig_path=None):
    # 此函数中直接读取块内容，与 dataclass 解析逻辑不完全相同，故结构变化不大
    # 核心依然是 defaultdict 收集数据
    history = defaultdict(list)
    full_time_list, success_flag_list = [], []
    success_count, failed_count = 0, 0

    for block in generate_epoch_blocks(filepath):
        block_content = block.replace("Current Epoch:", "", 1).strip()
        timestamp_match = re.match(r'^\s*([\d:\.]+)', block_content)
        
        if not timestamp_match: continue
            
        time = TimeSystem.TimeSystem().from_str('2020-09-01 ' + timestamp_match.group(1))
        full_time_list.append(time.datetime)

        if 'failed' in block:
            failed_count += 1
            success_flag_list.append(0)
            continue 
            
        success_count += 1
        success_flag_list.append(1)
        history['time'].append(time.datetime)
        
        pos_fixed = extract_array(block, 'Kinematics_Fixed:', shape='vector')
        err_vec = extract_array(block, 'Kinematics_Error:', shape='vector')
        
        if pos_fixed is not None:
            history['kin_pos'].append(pos_fixed[:3] / 1000.0)
            history['kin_clk'].append(pos_fixed[3] / CONST_C_ns)
        if err_vec is not None:
            history['kin_err'].append(err_vec * 100.0)

    for key in history:
        history[key] = np.array(history[key])

    if len(history['time']) > 0:
        time_arr = history['time']
        plot_state(time_arr, history['kin_pos'], ['X', 'Y', 'Z'], "Kinematics Position", "Position (km)", savefig_path=savefig_path)
        plot_state(time_arr, history['kin_err'], ['X_err', 'Y_err', 'Z_err'], "Kinematics Position Error", "Position Error(cm)", savefig_path=savefig_path, print_rmse=True, all_axis=True)

    plt.figure()
    plt.plot(full_time_list, success_flag_list, marker='o', linestyle='-', markersize=4)
    plt.title("Kinematics Success Flag Over Time")
    total = success_count + failed_count
    rate = (success_count / total * 100) if total > 0 else 0
    plt.text(0.5, 0.01, f'Success: {success_count}, Failed: {failed_count}, Rate: {rate:.2f}%', 
             ha='center', va='bottom', transform=plt.gca().transAxes, fontsize=10)
    plt.ylim(-0.1, 1.1)
    if savefig_path: plt.savefig(f"{savefig_path}/Kinematics_Success_Flag.png", dpi=300, bbox_inches='tight')
    plt.close()


def visualize_sats(epochs_data: List[EpochData], savefig_path=None, step=30):
    times, satset = [], set()
    for epoch in epochs_data:
        times.append(epoch.timestamp)
        satset.update(epoch.common_sats)
        
    if not times: return
    
    sat_list = sorted(list(satset))
    sat_to_index = {sat: i for i, sat in enumerate(sat_list)}
    
    plot_times, plot_sats = [], []
    dict_time_satnum = {}

    for ts, epoch in zip(times, epochs_data):
        sats = epoch.common_sats
        dict_time_satnum[ts] = len(sats)
        for sat in sats:
            if sat in sat_to_index:
                plot_times.append(ts.datetime)
                plot_sats.append(sat_to_index[sat])

    new_time_list, new_satnum_list = [], []
    for add_seconds in range(0, int((times[-1] - times[0])), step):
        current_time = times[0] + add_seconds
        new_time_list.append(current_time.datetime)
        closest_time = min(dict_time_satnum.keys(), key=lambda t: abs(t - current_time))
        new_satnum_list.append(0 if abs(closest_time - current_time) > step * 2 else dict_time_satnum[closest_time])

    plt.figure(figsize=(12, 8))
    plt.scatter(plot_times, plot_sats, s=20, alpha=0.8, marker='_')
    plt.yticks(ticks=range(len(satset)), labels=satset)
    plt.ylim(-1, len(satset))
    plt.ylabel('Satellite', fontsize=12)
    plt.xlabel('Time', fontsize=12)
    plt.title('GNSS Satellite Visibility Over Time', fontsize=14)
    if savefig_path: plt.savefig(f"{savefig_path}/GNSS_Satellite_Visibility.png", dpi=300, bbox_inches='tight')
    else: plt.show()


# ==========================================
# 7. 主程序执行入口 (Main Execution)
# ==========================================
import os

def split_log_by_itr(input_file_path, marker="Filter Iteration"):
    """
    按指定标记行拆分log文件，拆分文件保存在原文件同目录，返回新文件名称列表
    
    Args:
        input_file_path (str): 原始log文件路径
        marker (str): 分隔标志字符串（默认"Filter Iteration"）
    
    Returns:
        list: 拆分后生成的所有新文件的完整路径列表（若未拆分则返回空列表）
    """
    # 解析原始文件的路径、名称、后缀
    input_dir = os.path.dirname(input_file_path)  # 原始文件所在目录
    input_name = os.path.basename(input_file_path)  # 原始文件名（含后缀）
    name_without_ext, ext = os.path.splitext(input_name)  # 分离文件名和后缀
    
    # 初始化变量
    file_counter = 0  # 拆分文件的序号
    current_output_file = None  # 当前写入的文件对象
    new_files = []  # 存储生成的新文件路径列表
    
    try:
        # 以utf-8编码读取文件（兼容大多数log格式，若报错可尝试gbk）
        with open(input_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                # 检测到分隔标志行
                if marker in line:
                    # 关闭上一个输出文件（如果存在）
                    if current_output_file:
                        current_output_file.close()
                    
                    # 递增计数器，生成新文件名（原文件名_序号.原后缀）
                    file_counter += 1
                    output_file_name = f"{name_without_ext}_{file_counter}{ext}"
                    output_file_path = os.path.join(input_dir, output_file_name)
                    # 打开新文件并加入列表
                    current_output_file = open(output_file_path, 'w', encoding='utf-8')
                    new_files.append(output_file_path)
                    print(f"创建新文件：{output_file_path}（起始行：{line_num}）")
                
                # 如果当前有打开的输出文件，写入该行
                if current_output_file:
                    current_output_file.write(line)
        
        # 关闭最后一个输出文件
        if current_output_file:
            current_output_file.close()
        
        # 输出拆分结果
        if file_counter == 0:
            print(f"警告：在文件{input_file_path}中未找到标志'{marker}'，未拆分任何文件")
        else:
            print(f"\n拆分完成！共生成 {file_counter} 个文件")
            print(f"生成的文件列表：{new_files}")
    
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file_path}，请检查路径是否正确")
    except PermissionError:
        print(f"错误：没有权限读取{input_file_path}或写入{input_dir}")
    except Exception as e:
        print(f"未知错误：{str(e)}")
    finally:
        # 确保文件句柄被关闭
        if current_output_file and not current_output_file.closed:
            current_output_file.close()
    
    # 返回新文件列表
    return new_files


if __name__ == "__main__":
    starttime = None 
    endtime = None 
    # starttime = TimeSystem.TimeSystem().from_str('2020-09-11 00:20:00.000') 
    # endtime = TimeSystem.TimeSystem().from_str('2020-09-11 23:40:00.000')
    if os.path.exists('new'): shutil.rmtree('new')
    os.makedirs('new')

    flag0 = 'test_month'
    log_filepath = f"D:\\csu\\GNSS_MIX\\CSUAPPS\\CSUPODSApp\\BIN\\LOG\\Diff\\Diff_Log_YYYYMMDD_{flag0}.log"

    itr_files = split_log_by_itr(log_filepath, marker="Filter Iteration")

    for file_count, itr_file_path in enumerate(itr_files, 1):
        flag = f"{flag0}_{file_count}"
        files_path = split_file_by_marker(itr_file_path)

        for file_path in files_path:
            print(f"Processing file: {file_path}")
            mode = next((m for m in ['forward', 'backward', 'twoway', 'kinematics'] if m in file_path), '')

            epochs_data = parse_kalman_log_with_header(file_path, starttime=starttime, endtime=endtime)
            fig_path = os.path.join('new', f'{mode}_{flag}')
            os.makedirs(fig_path, exist_ok=True)
            
            if mode == 'twoway':
                parse_two_way_state(epochs_data[3:-1], savefig_path=fig_path)
            elif mode == 'kinematics':
                continue
                parse_kin_state(file_path, savefig_path=fig_path)
            else:
                parse_SC_state(epochs_data[3:-1], savefig_path=fig_path)
                visualize_sats(epochs_data[3:-1], savefig_path=fig_path)
        
        # RTN 坐标系残差提取
        log_filepath_twoway = f"D:\\csu\\GNSS_MIX\\CSUAPPS\\CSUPODSApp\\BIN\\LOG\\Diff\\Diff_Log_YYYYMMDD_{flag}_twoway.log"
        epochs_data = parse_kalman_log_with_header(log_filepath_twoway, starttime=starttime, endtime=endtime)

        fig_path_rtn = os.path.join('new', f'{flag}_RTN_twoway')
        os.makedirs(fig_path_rtn, exist_ok=True)
        
        rtn_history = defaultdict(list)
        for epoch in epochs_data:
            rtn_history['time'].append(epoch.timestamp.datetime)
            r_j2000, v_j2000 = epoch.state_vector[0:3], epoch.state_vector[3:6]
            rtn_history['pos_err'].append(j2000_to_rtn(r_j2000, v_j2000, epoch.smoothed_sc_b_state_err[0:3]))
            rtn_history['vel_err'].append(j2000_to_rtn(r_j2000, v_j2000, epoch.smoothed_sc_b_state_err[3:6]))
            rtn_history['rts_pos_res'].append(j2000_to_rtn(r_j2000, v_j2000, epoch.rts_error[0:3]))
            rtn_history['rts_vel_res'].append(j2000_to_rtn(r_j2000, v_j2000, epoch.rts_error[3:6]))
            
        plot_state(rtn_history['time'], np.array(rtn_history['pos_err']) * 1000, ['Radial', 'Transverse', 'Normal'], 
                "Position Residuals", "Position Residuals (mm)", savefig_path=fig_path_rtn, print_rmse=True, all_axis=True, savecsv=True)
        plot_state(rtn_history['time'], np.array(rtn_history['vel_err']) * 1000, ['Radial', 'Transverse', 'Normal'], 
                "Velocity Residuals", "Velocity Residuals (mm/s)", savefig_path=fig_path_rtn, print_rmse=True, all_axis=True, savecsv=True)
        plot_state(rtn_history['time'], np.array(rtn_history['rts_pos_res']) * 1000, ['Radial', 'Transverse', 'Normal'], 
                "RTS Position Residuals", "RTS Position Residuals (mm)", savefig_path=fig_path_rtn, print_rmse=True, all_axis=True, savecsv=True)
        plot_state(rtn_history['time'], np.array(rtn_history['rts_vel_res']) * 1000, ['Radial', 'Transverse', 'Normal'], 
                "RTS Velocity Residuals", "RTS Velocity Residuals (mm/s)", savefig_path=fig_path_rtn, print_rmse=True, all_axis=True, savecsv=True)

        log_filepath_twoway = f"D:\\csu\\GNSS_MIX\\CSUAPPS\\CSUPODSApp\\BIN\\LOG\\Diff\\Diff_Log_YYYYMMDD_{flag}_forward.log"
        epochs_data = parse_kalman_log_with_header(log_filepath_twoway, starttime=starttime, endtime=endtime)

        fig_path_rtn = os.path.join('new', f'{flag}_RTN_forward')
        os.makedirs(fig_path_rtn, exist_ok=True)
        
        rtn_history = defaultdict(list)
        for epoch in epochs_data:
            rtn_history['time'].append(epoch.timestamp.datetime)
            r_j2000, v_j2000 = epoch.state_vector[0:3], epoch.state_vector[3:6]
            rtn_history['pos_err'].append(j2000_to_rtn(r_j2000, v_j2000, epoch.sc_b_state_err[0:3]))
            rtn_history['vel_err'].append(j2000_to_rtn(r_j2000, v_j2000, epoch.sc_b_state_err[3:6]))
            
        plot_state(rtn_history['time'], np.array(rtn_history['pos_err']) * 1000, ['Radial', 'Transverse', 'Normal'], 
                "Position Residuals", "Position Residuals (mm)", savefig_path=fig_path_rtn, print_rmse=True, all_axis=True, savecsv=True)
        plot_state(rtn_history['time'], np.array(rtn_history['vel_err']) * 1000, ['Radial', 'Transverse', 'Normal'], 
                "Velocity Residuals", "Velocity Residuals (mm/s)", savefig_path=fig_path_rtn, print_rmse=True, all_axis=True, savecsv=True)


        log_filepath_twoway = f"D:\\csu\\GNSS_MIX\\CSUAPPS\\CSUPODSApp\\BIN\\LOG\\Diff\\Diff_Log_YYYYMMDD_{flag}_backward.log"
        epochs_data = parse_kalman_log_with_header(log_filepath_twoway, starttime=starttime, endtime=endtime)

        fig_path_rtn = os.path.join('new', f'{flag}_RTN_backward')
        os.makedirs(fig_path_rtn, exist_ok=True)
        
        rtn_history = defaultdict(list)
        for epoch in epochs_data:
            rtn_history['time'].append(epoch.timestamp.datetime)
            r_j2000, v_j2000 = epoch.state_vector[0:3], epoch.state_vector[3:6]
            rtn_history['pos_err'].append(j2000_to_rtn(r_j2000, v_j2000, epoch.sc_b_state_err[0:3]))
            rtn_history['vel_err'].append(j2000_to_rtn(r_j2000, v_j2000, epoch.sc_b_state_err[3:6]))
            
        plot_state(rtn_history['time'], np.array(rtn_history['pos_err']) * 1000, ['Radial', 'Transverse', 'Normal'], 
                "Position Residuals", "Position Residuals (mm)", savefig_path=fig_path_rtn, print_rmse=True, all_axis=True, savecsv=True)
        plot_state(rtn_history['time'], np.array(rtn_history['vel_err']) * 1000, ['Radial', 'Transverse', 'Normal'], 
                "Velocity Residuals", "Velocity Residuals (mm/s)", savefig_path=fig_path_rtn, print_rmse=True, all_axis=True, savecsv=True)
