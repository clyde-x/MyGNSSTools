import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil

import re
import numpy as np
from collections import defaultdict

import TimeSystem

CONST_C = 299792458.0  # 光速，单位：m/s
CONST_C_ns = CONST_C * 1e-9  # 光速，单位：m/ns

def extract_array(block, label, shape=None, dtype=float):
    # 构造正则表达式：匹配 "Label, size: N" 或 "Label, size: NxM"
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
        # 提取非空白字符串 token（可跨行，按空白分隔）
        
        tokens = tail[1:].split('\t')  # split() 默认按任意空白分割，包括换行、空格、tab
        tokens = [token.strip() for token in tokens if token.strip() != '']
        if len(tokens) < expected_size:
            print(f"Warning: Not enough tokens for '{label}' (need {expected_size}, got {len(tokens)})")
            return None
        # 返回字符串数组
        return np.array(tokens[:expected_size], dtype=str).tolist()

    else:
        # 从匹配位置之后提取所有浮点数（支持科学计数法）
        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", tail)

        if len(numbers) < expected_size:
            print(f"Warning: Not enough numbers for '{label}' (need {expected_size}, got {len(numbers)})")
            return None

        arr = np.array(numbers[:expected_size], dtype=dtype)
        if shape == 'matrix':
            return arr.reshape((rows, cols))
        else:
            return arr

def extract_lambda_info(block, last_lambda):
    lambda_dict = {}
    # dict_keys: fixed - bool; prob, sqn_s, sqn_m
    pattern = r'LAMBDA fix \(L1\+L2\):\s*(OK|NO)\b'

    lambda_dict['fixed'] = True
    match = re.search(pattern, block)
    if match:
        if match.group(1) == 'NO':
            lambda_dict['fixed'] = False 
    
    
    pattern = r'prob=(?P<prob>[-+]?\d+\.\d+e[-+]\d+), sqn_s=(?P<sqn_s>[-+]?\d+\.\d+e[-+]\d+), sqn_m=(?P<sqn_m>[-+]?\d+\.\d+e[-+]\d+)'
    match = re.search(pattern, block)
    if match:
        prob = float(match.group('prob'))
        sqn_s = float(match.group('sqn_s'))
        sqn_m = float(match.group('sqn_m'))
    
    else:
        prob = last_lambda.get('prob', np.nan)
        sqn_s = last_lambda.get('sqn_s', np.nan)
        sqn_m = last_lambda.get('sqn_m', np.nan)
    
    lambda_dict['prob'] = prob
    lambda_dict['sqn_s'] = sqn_s
    lambda_dict['sqn_m'] = sqn_m

    return lambda_dict


def RMSE(residuals):
    if residuals.size == 0:
        return np.array([])

    squared_residuals = residuals ** 2
    mean_squared_residuals = np.mean(squared_residuals, axis=0)
    rmse = np.sqrt(mean_squared_residuals)
    return rmse

def parse_kalman_log_with_header(filepath, starttime=None, endtime=None):
    if starttime is not None and type(starttime) != TimeSystem.TimeSystem:
        raise ValueError("starttime must be a TimeSystem.TimeSystem object or None.")
    if endtime is not None and type(endtime) != TimeSystem.TimeSystem:
        raise ValueError("endtime must be a TimeSystem.TimeSystem object or None.")

    with open(filepath, 'r') as f:
        content = f.read()

    # 找到第一个 "Current Epoch:" 的位置，丢弃之前的内容（即文件头）
    epoch_start = content.find("Current Epoch:")
    if epoch_start == -1:
        raise ValueError("No 'Current Epoch:' found in the log file.")
    
    # 从第一个 Epoch 开始截取
    content = content[epoch_start:]

    # 按 "Current Epoch:" 切分（注意：第一个块现在以 "Current Epoch:" 开头）
    epoch_blocks = content.split("Current Epoch:")
    # 第一个元素是空字符串或空白（因为 split 会在开头产生一个空块），跳过
    if not epoch_blocks[0].strip():
        epoch_blocks = epoch_blocks[1:]

    epochs_data = []
    last_lambda = {'fixed': False}
    for block in epoch_blocks:
        epoch_info = {}
        # 提取时间戳
        timestamp_match = re.match(r'^\s*([\d:\.]+)', block)
        time = TimeSystem.TimeSystem()
        if timestamp_match:
            time = time.from_str('2020-09-01 '+timestamp_match.group(1))
            if starttime is not None and time < starttime:
                continue
            if endtime is not None and time > endtime:
                continue
        epoch_info['Timestamp'] = time

        # 提取所有字段
        epoch_info_vector_keys = ["Spacecraft A State", "Spacecraft B State", "State Vector", "Spacecraft B Ground Truth State", "Spacecraft B State Error (GT - Est)",
                                  "Predicted State Vector", "Predicted Residuals", 'Updated State Vector', 'Post-fit Residuals', 'Smoothed State Vector', 'Smoothed Spacecraft B State Error (GT - Est)',
                                  'Satellites with Cycle Slips']
        epoch_info_matrix_keys = ["Covariance Matrix", "State Transition Matrix", "Process Noise Matrix",
                                  "Predicted Covariance Matrix", "Design Matrix H", "Kalman Gain K", "Updated Covariance Matrix"]
        for key in epoch_info_vector_keys:
            epoch_info[key] = extract_array(block, key, shape='vector')
        for key in epoch_info_matrix_keys:
            epoch_info[key] = extract_array(block, key, shape='matrix')
        epoch_info['Common Satellites'] = extract_array(block, "Common Satellites", shape='SatSet')
        last_lambda = extract_lambda_info(block, last_lambda)
        epoch_info['LAMBDA'] = last_lambda
        epochs_data.append(epoch_info)

    return epochs_data

import numpy as np

def create_measurement_noise_covariance(uiNumSats, usePhasae):
    """
    创建测量噪声协方差矩阵 R
    """
    uiNumObs = uiNumSats * (4 if usePhasae else 2)

    R = np.zeros((uiNumObs, uiNumObs))

    code_var = 5 ** 2
    phase_var = 0.05 ** 2

    for i in range(uiNumSats):
        R[i, i] = code_var
        R[i + uiNumSats, i + uiNumSats] = code_var

        if usePhasae:
            R[i + 2 * uiNumSats, i + 2 * uiNumSats] = phase_var
            R[i + 3 * uiNumSats, i + 3 * uiNumSats] = phase_var

    return R

def matrix_heat_map(data, title, xlabel, ylabel, row_labels=None, col_labels=None, savefig_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(data, cmap='viridis', cbar_kws={'label': 'Value'},
        xticklabels=col_labels if col_labels is not None else 'auto',
        yticklabels=row_labels if row_labels is not None else 'auto')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if savefig_path is not None:
        plt.savefig(f"{savefig_path}/{title.replace(' ', '_')}_heat.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()

def matrix_one_zero_map(data, title, xlabel, ylabel, row_labels=None, col_labels=None, savefig_path=None):
    from matplotlib.colors import ListedColormap
    eps = 1e-8

    plt.figure(figsize=(8, 6))
    data = (np.abs(data) >= eps).astype(int)
    
    # 定义 colormap：0 -> white, 1 -> gray
    cmap = ListedColormap(['white', 'gray'])
    
    # 绘制热力图
    sns.heatmap(data, cmap=cmap,square=True, cbar=False,
        linewidths=0.5, linecolor='black',
        xticklabels=col_labels if col_labels is not None else 'auto',
        yticklabels=row_labels if row_labels is not None else 'auto'
    )
    plt.text(1, 1, 'threshold: {}'.format(eps), ha='center', va='center', transform=plt.gca().transAxes)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if savefig_path is not None:
        plt.savefig(f"{savefig_path}/{title.replace(' ', '_')}_01.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()

def plot_state(time_list, data_list, name_list, title, ylabel, scatter = False, savefig_path=None, print_rmse=False, print_mean_std=False, cs_time_list = []):
    plt.figure(figsize=(10, 6))
    if data_list.shape[1] != len(name_list):
        raise ValueError("Data list length and name list length do not match.")
    for i in range(data_list.shape[1]):
        if scatter:
            plt.scatter(time_list, data_list[:, i], label=name_list[i])
        else:
            plt.plot(time_list, data_list[:, i], label=name_list[i])
    if print_rmse:
        for i in range(data_list.shape[1]):
            rmse = RMSE(data_list[:, i])
            print(f"RMSE of {name_list[i]}: {rmse}",end = ', ')
        print()
        #print rmse on the fig
        plt.text(0.5, 0.01, 'RMSE: ' + ', '.join([f"{name_list[i]}: {RMSE(data_list[:, i]):.4f}" for i in range(data_list.shape[1])]), 
                 ha='center', va='bottom', transform=plt.gca().transAxes, fontsize=10)
        
    if print_mean_std:
        for i in range(data_list.shape[1]):
            mean = np.mean(data_list[:, i])
            std = np.std(data_list[:, i])
            print(f"Mean and Std of {name_list[i]}: {mean}, {std}", end=', ')
        print()
        # print mean and std on the fig
        plt.text(0.5, 0.01, 'Mean & Std: ' + ', '.join([f"{name_list[i]}: {np.mean(data_list[:, i]):.4f} ± {np.std(data_list[:, i]):.4f}" for i in range(data_list.shape[1])]), 
                 ha='center', va='bottom', transform=plt.gca().transAxes, fontsize=10)
    
    for cs_time in cs_time_list:
        plt.axvline(x=cs_time.datetime, color='r', linestyle='--', alpha=0.5)
    
        
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.legend()
    if savefig_path is not None:
        plt.savefig(f"{savefig_path}/{title.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def parse_SC_state(epochs_data, savefig_path=None, Cr=True, Cd=True, Acc=True):
    basic_state = 6
    dyn_state = 0
    if Cr:
        dyn_state += 1
        cr_index = basic_state + dyn_state - 1
    if Cd:
        dyn_state += 1
        cd_index = basic_state + dyn_state - 1
    if Acc:
        dyn_state += 3
        acc_index = basic_state + dyn_state - 3
    clk_index = basic_state + dyn_state
    time_list = []
    Apos_list = []
    Avel_list = []
    Aforce_list = []
    Aacc_list = []
    
    Bpos_list = []
    Bvel_list = []
    Bforce_list = []
    Bacc_list = []
    Bpos_gt_list = []
    Bpos_res_list = []
    Bvel_gt_list = []
    Bvel_res_list = []
    post_res_list = []

    relative_pos_list = []
    relative_vel_list = []
    relative_cdcr_list = []
    relative_acc_list = []
    relative_clk_list = []

    cs_time_list = []

    fiexd_list = []
    lambda_info_list = []


    #iono_list = []
    for epoch_data in epochs_data:
        time_list.append(epoch_data['Timestamp'].datetime)
        Astate = epoch_data['Spacecraft A State']
        Bstate = epoch_data['Spacecraft B State']
        Apos_list.append(Astate[0:3]/1000.0)  # 转换为 km
        Avel_list.append(Astate[3:6])
        if Cr and Cd:
            Aforce_list.append(Astate[cr_index:cd_index+1])
        else:
            Aforce_list.append([np.nan, np.nan])
        if Acc:
            Aacc_list.append(Astate[acc_index:acc_index+3]*1e9)  # 转换为 nm/s²
        else:
            Aacc_list.append([np.nan, np.nan, np.nan])
        Bpos_list.append(Bstate[0:3]/1000.0)  # 转换为 km
        Bvel_list.append(Bstate[3:6])
        if Cr and Cd:
            Bforce_list.append(Bstate[cr_index:cd_index+1])
        else:
            Bforce_list.append([np.nan, np.nan])
        if Acc:
            Bacc_list.append(Bstate[acc_index:acc_index+3]*1e9)  # 转换为 nm/s²
        else:
            Bacc_list.append([np.nan, np.nan, np.nan])
        BState_gt = epoch_data['Spacecraft B Ground Truth State']
        BState_res = epoch_data['Spacecraft B State Error (GT - Est)']
        post_res = epoch_data['Post-fit Residuals']
        Bpos_gt_list.append(BState_gt[0:3]/1000.0)
        Bpos_res_list.append(BState_res[0:3])
        Bvel_gt_list.append(BState_gt[3:6])
        Bvel_res_list.append(BState_res[3:6])
        post_res_list.append(RMSE(post_res))

        state = epoch_data['State Vector']
        relative_pos_list.append(state[0:3]/1000.0)  # 转换为 km
        relative_vel_list.append(state[3:6])
        if Cr and Cd:
            relative_cdcr_list.append(state[cr_index:cd_index+1])
        else:
            relative_cdcr_list.append([np.nan, np.nan])
        if Acc:
            relative_acc_list.append(state[acc_index:acc_index+3]*1e9)  # 转换为 nm/s²
        else:
            relative_acc_list.append([np.nan, np.nan, np.nan])
        relative_clk_list.append(state[clk_index])

        if epoch_data.get('Satellites with Cycle Slips') is not None:
            cs_time_list.append(epoch_data['Timestamp'])
        
        #iono_list.append(state[12:12+len(epoch_data['Common Satellites'])])
        if epoch_data['LAMBDA']['fixed']: 
            fiexd_list.append(1)
        else:
            fiexd_list.append(0)
        lambda_info_list.append(epoch_data['LAMBDA'])

    
    plot_state(time_list, np.array(Apos_list), ['X', 'Y', 'Z'], "Spacecraft A Position", "Position (km)", savefig_path=savefig_path)
    plot_state(time_list, np.array(Avel_list), ['Vx', 'Vy', 'Vz'], "Spacecraft A Velocity", "Velocity (m/s)", savefig_path=savefig_path)
    if Cr and Cd:
        plot_state(time_list, np.array(Aforce_list), ['Cd', 'Cr'], "Spacecraft A Force Coefficients", "Cd and Cr", savefig_path=savefig_path)
    if Acc:
        plot_state(time_list, np.array(Aacc_list), ['ar', 'at', 'an'], "Spacecraft A Accelerations", "EmpAcc (nm/s²)", savefig_path=savefig_path)
    plot_state(time_list, np.array(Bpos_list), ['X', 'Y', 'Z'], "Spacecraft B Position", "Position (km)", savefig_path=savefig_path)
    plot_state(time_list, np.array(Bvel_list), ['Vx', 'Vy', 'Vz'], "Spacecraft B Velocity", "Velocity (m/s)", savefig_path=savefig_path)
    if Cr and Cd:
        plot_state(time_list, np.array(Bforce_list), ['Cd', 'Cr'], "Spacecraft B Force Coefficients", "Cd and Cr", savefig_path=savefig_path)
    if Acc:
        plot_state(time_list, np.array(Bacc_list), ['ar', 'at', 'an'], "Spacecraft B Accelerations", "EmpAcc (nm/s²)", savefig_path=savefig_path)
    plot_state(time_list, np.array(relative_clk_list).reshape(-1, 1)/CONST_C_ns, ['Clock Bias'], "Spacecraft Clock Bias", "Clock Bias (ns)", savefig_path=savefig_path, print_mean_std = True, cs_time_list=cs_time_list)
    #plot_state(time_list, np.array(iono_list), [f'Iono_{i+1}' for i in range(len(epochs_data[0]['Common Satellites']))], "Ionospheric Delays", "Ionospheric Delay (m)", savefig_path=savefig_path)
    plot_state(time_list, np.array(Bpos_gt_list), ['X_gt', 'Y_gt', 'Z_gt'], "Spacecraft B Ground Truth Position", "Position (km)", savefig_path=savefig_path)
    plot_state(time_list, np.array(Bpos_res_list)*100, ['X_res', 'Y_res', 'Z_res'], "Spacecraft B Position Residuals (GT - Est)", "Position Residuals (cm)", savefig_path=savefig_path, print_rmse=True, cs_time_list=cs_time_list)
    plot_state(time_list, np.array(Bvel_gt_list), ['Vx_gt', 'Vy_gt', 'Vz_gt'], "Spacecraft B Ground Truth Velocity", "Velocity (m/s)", savefig_path=savefig_path)
    plot_state(time_list, np.array(Bvel_res_list)*100, ['Vx_res', 'Vy_res', 'Vz_res'], "Spacecraft B Velocity Residuals (GT - Est)", "Velocity Residuals (cm/s)", savefig_path=savefig_path, print_rmse=True, cs_time_list=cs_time_list)
    plot_state(time_list, np.array(post_res_list).reshape(-1, 1), ['post residual'], "Post-fit Residuals(RMSE)", "Residuals (m)", savefig_path=savefig_path)
    plot_state(time_list, np.array(relative_pos_list), ['X_AB', 'Y_AB', 'Z_AB'], "Relative Position (A-B)", "Relative Position (km)", savefig_path=savefig_path)
    plot_state(time_list, np.array(relative_vel_list), ['Vx_AB', 'Vy_AB', 'Vz_AB'], "Relative Velocity (A-B)", "Relative Velocity (m/s)", savefig_path=savefig_path)
    if Cr and Cd:
        plot_state(time_list, np.array(relative_cdcr_list), ['Cd_AB', 'Cr_AB'], "Relative Force Coefficients (A-B)", "Relative Cd and Cr", savefig_path=savefig_path)
    if Acc:
        plot_state(time_list, np.array(relative_acc_list), ['ar_AB', 'at_AB', 'an_AB'], "Relative Accelerations (A-B)", "Relative EmpAcc (nm/s²)", savefig_path=savefig_path)

    plot_state(time_list, np.array(fiexd_list).reshape(-1, 1), ['LAMBDA Fixed'], "LAMBDA Fix Flag", "Fixed Flag", savefig_path=savefig_path, scatter=True, print_mean_std=True, cs_time_list=cs_time_list)
    print("fiexd count: ", np.sum(fiexd_list), "total count: ", len(fiexd_list), "fixed rate: ", np.sum(fiexd_list)/len(fiexd_list))

def parse_two_way_state(epochs_data, savefig_path=None, Cr=True, Cd=True, Acc=True):
    basic_state = 6
    dyn_state = 0
    if Cd:
        dyn_state += 1
        cd_index = basic_state + dyn_state - 1
    if Cr:
        dyn_state += 1
        cr_index = basic_state + dyn_state - 1
    if Acc:
        dyn_state += 3
        acc_index = basic_state + dyn_state - 3
    clk_index = basic_state + dyn_state
    time_list = []
    state_list = []
    pos_list = []
    vel_list = []
    force_list = []
    acc_list = []
    clk_list = []
    pos_res_list = []
    vel_res_list = []

    cs_time_list = []

    for epoch_data in epochs_data:
        time_list.append(epoch_data['Timestamp'].datetime)
        state = epoch_data['State Vector']
        residuals = epoch_data['Smoothed Spacecraft B State Error (GT - Est)']

        pos_list.append(state[0:3]/1000.0)  # 转换为 km
        pos_res_list.append(residuals[0:3]*100.0)  # 转换为 cm
        
        vel_list.append(state[3:6])
        vel_res_list.append(residuals[3:6]*1000.0)  # 转换为 cm/s

        if Cd and Cr:
            force_list.append(state[cd_index:cr_index+1])
        else:
            force_list.append([np.nan, np.nan])
        if Acc:
            acc_list.append(state[acc_index:acc_index+3]*1e9)  # 转换为 nm/s²
        else:
            acc_list.append([np.nan, np.nan, np.nan])
        clk_list.append(state[clk_index])

        if epoch_data.get('Satellites with Cycle Slips') is not None:
            cs_time_list.append(epoch_data['Timestamp'])
    CONST_C = 299792458.0  # 光速，单位：m/s
    CONST_C_ns = CONST_C * 1e-9  # 光速，单位：m/ns
    plot_state(time_list, np.array(pos_list), ['X_AB', 'Y_AB', 'Z_AB'], "Two-Way Relative Position (A-B)", "Relative Position (km)", savefig_path=savefig_path)
    plot_state(time_list, np.array(vel_list), ['Vx_AB', 'Vy_AB', 'Vz_AB'], "Two-Way Relative Velocity (A-B)", "Relative Velocity (m/s)", savefig_path=savefig_path)
    if Cd and Cr:
        plot_state(time_list, np.array(force_list), ['Cd_AB', 'Cr_AB'], "Two-Way Relative Force Coefficients (A-B)", "Relative Cd and Cr", savefig_path=savefig_path)
    if Acc:
        plot_state(time_list, np.array(acc_list), ['ar_AB', 'at_AB', 'an_AB'], "Two-Way Relative Accelerations (A-B)", "Relative EmpAcc (nm/s²)", savefig_path=savefig_path)
    plot_state(time_list, np.array(clk_list).reshape(-1, 1)/CONST_C_ns, ['Clock Bias'], "Two-Way Spacecraft Clock Bias", "Clock Bias (ns)", savefig_path=savefig_path, print_mean_std = True, cs_time_list=cs_time_list)

    plot_state(time_list, np.array(pos_res_list), ['X_res', 'Y_res', 'Z_res'], "Two-Way Relative Position Residuals (GT - Est)", "Position Residuals (cm)", savefig_path=savefig_path, print_rmse=True, cs_time_list=cs_time_list)
    plot_state(time_list, np.array(vel_res_list), ['Vx_res', 'Vy_res', 'Vz_res'], "Two-Way Relative Velocity Residuals (GT - Est)", "Velocity Residuals (mm/s)", savefig_path=savefig_path, print_rmse=True, cs_time_list=cs_time_list)

def parse_kin_state(filepath, savefig_path=None):
    with open(filepath, 'r') as f:
        content = f.read()

    epoch_start = content.find("Current Epoch:")
    if epoch_start == -1:
        raise ValueError("No 'Current Epoch:' found in the log file.")

    content = content[epoch_start:]

    epoch_blocks = content.split("Current Epoch:")
    if not epoch_blocks[0].strip():
        epoch_blocks = epoch_blocks[1:]

    epochs_data = []
    failed_couny = 0
    success_count = 0
    time_list = []
    success_flag_list = []
    full_time_list = []
    kin_pos_list = []
    kin_clk_list = []
    kin_err_list = []
    for block in epoch_blocks:
        epoch_info = {}
        timestamp_match = re.match(r'^\s*([\d:\.]+)', block)
        time = TimeSystem.TimeSystem()
        if timestamp_match:
            time = time.from_str('2020-09-01 '+timestamp_match.group(1))
        epoch_info['Timestamp'] = time
        full_time_list.append(time.datetime)

        if 'failed' in block:
            failed_couny += 1
            success_flag_list.append(int(0))
            continue 
        else:
            success_count += 1
            success_flag_list.append(int(1))
            time_list.append(time.datetime)
            epoch_vec_keys = ['Kinematics_Fixed:', 'Kinematics_Error:']
            for key in epoch_vec_keys:
                epoch_info[key] = extract_array(block, key, shape='vector')
            kin_pos_list.append(epoch_info['Kinematics_Fixed:'][:3]/1000.0)  # 转换为 km
            kin_clk_list.append(epoch_info['Kinematics_Fixed:'][3])
            kin_err_list.append(epoch_info['Kinematics_Error:'])
            epochs_data.append(epoch_info)
    
    plot_state(time_list, np.array(kin_pos_list), ['X', 'Y', 'Z'], "Kinematics Position", "Position (km)", savefig_path=savefig_path)
    plot_state(time_list, np.array(kin_err_list)*100, ['X_err', 'Y_err', 'Z_err'], "Kinematics Position Error", "Position Error(cm)", savefig_path=savefig_path, print_rmse=True)
    plot_state(time_list, np.array(kin_clk_list).reshape(-1, 1)/CONST_C_ns, ['clk'], "Kinematics Clock Bias", "Clock Bias (ns)", savefig_path=savefig_path, print_mean_std=True)
    plt.close()

    plt.plot(full_time_list, success_flag_list, marker='o', linestyle='-', markersize=4)
    plt.title("Kinematics Success Flag Over Time")
    plt.text(0.5, 0.01, f'Success Count: {success_count}, Failed Count: {failed_couny}, Success Rate: {success_count/(success_count+failed_couny)*100:.2f}%', 
             ha='center', va='bottom', transform=plt.gca().transAxes, fontsize=10)
    plt.ylim(-0.1, 1.1)
    plt.yticks([0, 1], ['Fail', 'Success'])
    plt.xlabel("Time")
    plt.ylabel("Kinematics Success Flag")
    if savefig_path is not None:
        plt.savefig(f"{savefig_path}/Kinematics_Success_Flag.png", dpi=300, bbox_inches='tight')
    plt.close()


def visualize_sats(epochs_data, savefig_path=None, step = 30):
    times  = []
    satset = set()
    for epoch_data in epochs_data:
        times.append(epoch_data['Timestamp'])
        sats = epoch_data.get('Common Satellites', [])
        satset.update(sats)
    sat_list = sorted(list(satset))
    sat_to_index = {sat: i for i, sat in enumerate(sat_list)}
    # 构建可视化点
    plot_times = []
    plot_sats = []
    dict_time_satnum = {}
    time_step_list = []
    last_time = TimeSystem.TimeSystem()
    last_time.from_str('2020-09-01 00:00:00')

    for ts, epoch in zip(times, epochs_data):
        sats = epoch.get('Common Satellites', [])
        dict_time_satnum[ts] = len(sats)
        for sat in sats:
            if sat in sat_to_index:
                plot_times.append(ts.datetime)
                plot_sats.append(sat_to_index[sat])
        time_step_list.append(ts-last_time)
        last_time = ts


    new_time_list = []
    new_satnum_list = []
    for add_seconds in range(0, int((times[-1]-times[0])), step):
        current_time = times[0] + add_seconds
        new_time_list.append(current_time.datetime)
        # 在 dict_time_satnum 中查找最接近 current_time 的时间点
        closest_time = min(dict_time_satnum.keys(), key=lambda t: abs(t - current_time))
        if abs(closest_time - current_time) > step*2:
            new_satnum_list.append(0)
        else:
            new_satnum_list.append(dict_time_satnum[closest_time])

    # 绘图
    plt.figure(figsize=(12, 8))
    plt.scatter(plot_times, plot_sats, s=20, alpha=0.8, marker='_')

    # 设置纵轴：卫星名称
    plt.yticks(ticks=range(len(satset)), labels=satset)
    plt.ylim(-1, len(satset))
    plt.ylabel('Satellite', fontsize=12)

    # 设置横轴：时间
    plt.xlabel('Time', fontsize=12)
    plt.title('GNSS Satellite Visibility Over Time', fontsize=14)

    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    if savefig_path is not None:
        plt.savefig(f"{savefig_path}/GNSS_Satellite_Visibility.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(new_time_list, new_satnum_list, linestyle='-', markersize=4)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Number of Visible Satellites', fontsize=12)
    plt.title('Number of Visible GNSS Satellites Over Time', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if savefig_path is not None:
        plt.savefig(f"{savefig_path}/GNSS_Satellite_Count.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()

    



def split_file_by_marker(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    marker1 = "*** backward filtering ***    \n"  # 注意换行符
    marker2 = "Start Smoothing.\n"
    marker3 = "Start Kinematics.\n"
    try:
        idx1 = lines.index(marker1)
    except ValueError:
        print("错误：未找到分界行 '*** backward filtering ***'")
        return [input_file]
    try:
        idx2 = lines.index(marker2)
    except ValueError:
        print("错误：未找到分界行 'Start Smoothing.'")
        idx2 = -1
    try:
        idx3 = lines.index(marker3)
    except ValueError:
        print("错误：未找到分界行 'Start Kinematics.'")
        idx3 = -1

    base_name, ext = os.path.splitext(input_file)
    forward_file = f"{base_name}_forward{ext}"
    backward_file = f"{base_name}_backward{ext}"
    twoway_file = f"{base_name}_twoway{ext}"
    kinematics_file = f"{base_name}_kinematics{ext}"

    # 写入 forward 文件（marker1 之前的内容）
    with open(forward_final := forward_file, 'w', encoding='utf-8') as f:
        f.writelines(lines[:idx1])

    # 写入 backward 文件（从 marker1 开始到 marker2）
    with open(backward_file, 'w', encoding='utf-8') as f:
        f.writelines(lines[idx1:idx2])

    # 写入 twoway 文件（从 marker 之后到末尾）
    with open(twoway_file, 'w', encoding='utf-8') as f:
        f.writelines(lines[idx2:idx3])

    if idx3 != -1:
        # 写入 kinematics 文件（从 marker3 到末尾）
        with open(kinematics_file, 'w', encoding='utf-8') as f:
            f.writelines(lines[idx3:])

    print(f"已生成：{forward_file} 和 {backward_file} 和 {twoway_file} 和 {kinematics_file}")
    return [forward_file, backward_file, twoway_file, kinematics_file]


def construct_labels(state_list, label_lenth, sat_list=None, usephase=False):
    # input: sate_list: ['X', 'Y', 'Z', 'vx', 'vy', 'vz', 'Cd', 'Cr', 'ar', 'at', 'an', 'clk']
    # input: label_lenth: int, total length of labels, usually the shape[0] of matrix
    # function: construct the full label list including satellite related states, include iono and amb states
    sat_num = label_lenth - len(state_list)
    sat_num = int(sat_num // (3 if usephase else 1))
    if sat_list is None:
        sat_list = [' ' * sat_num for _ in range(sat_num)]
    labels = state_list.copy()
    for sat in sat_list:
        labels.append(f'Iono_{sat}')
    if usephase:
        for sat in sat_list:
            labels.append(f'Amb1_{sat}')
        for sat in sat_list:
            labels.append(f'Amb2_{sat}')
    return labels
def construct_observation_labels(sat_list, obs_type_list):
    labels = []
    for obs_type in obs_type_list:
        for sat in sat_list:
            labels.append(f'{obs_type}_{sat}')
    return labels


def visualize_matrix(epochs_data, epoch=1, obs_types=['C1', 'P2']):
    epoch_data = epochs_data[epoch]

    SCA_state = epoch_data['Spacecraft A State']
    SCB_state = epoch_data['Spacecraft B State']
    state = epoch_data['State Vector']
    P_last = epoch_data['Covariance Matrix']
    Phi = epoch_data['State Transition Matrix']
    Q = epoch_data['Process Noise Matrix']
    satset = epoch_data['Common Satellites']
    state_minus = epoch_data['Predicted State Vector']
    P_minus = epoch_data['Predicted Covariance Matrix']
    Res_minus = epoch_data['Predicted Residuals']
    H_design = epoch_data['Design Matrix H']
    K_gain = epoch_data['Kalman Gain K']
    state_plus = epoch_data['Updated State Vector']
    P_plus = epoch_data['Updated Covariance Matrix']
    print(epoch_data['Timestamp'].to_digital())

    y_name_list = ['X', 'Y', 'Z', 'Vx', 'Vy', 'Vz', 'Cd', 'Cr', 'ar', 'at', 'an', 'clk']
    state_label = construct_labels(y_name_list, K_gain.shape[0], sat_list=satset, usephase=False)
    obs_label = construct_observation_labels(satset, obs_types)

    matrix_heat_map(K_gain, "Kalman Gain K", "Columns", "Rows", col_labels=obs_label, row_labels=state_label)
    fig_path = 'FilterVisualizeImgs' + f'{epoch}_' + f'{epoch_data["Timestamp"].to_digital()}'
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
        print(f"Created directory: {fig_path}")
    
    matrix_one_zero_map(Phi, "Phi State Transition Matrix", " ", " ", col_labels=state_label, row_labels=state_label, savefig_path = fig_path)
    matrix_one_zero_map(Q, "Q Process Noise Matrix", " ", " ", col_labels=state_label, row_labels=state_label, savefig_path = fig_path)
    matrix_one_zero_map(K_gain, "K Kalman Gain", " ", " ", col_labels=obs_label, row_labels=state_label, savefig_path = fig_path)
    matrix_one_zero_map(P_minus, "P- Predicted Covariance Matrix", " ", " ", col_labels=state_label, row_labels=state_label, savefig_path = fig_path)

    matrix_one_zero_map(H_design, "H Design Matrix", " ", " ", col_labels=state_label, row_labels=obs_label, savefig_path = fig_path)
    matrix_one_zero_map(P_plus, "P+ Updated Covariance Matrix", " ", " ", col_labels=state_label, row_labels=state_label, savefig_path = fig_path)
    
    matrix_heat_map(Phi, "Phi State Transition Matrix", " ", " ", col_labels=state_label, row_labels=state_label, savefig_path = fig_path)
    matrix_heat_map(Q, "Q Process Noise Matrix", " ", " ", col_labels=state_label, row_labels=state_label, savefig_path = fig_path)
    matrix_heat_map(K_gain, "K Kalman Gain", " ", " ", col_labels=obs_label, row_labels=state_label, savefig_path = fig_path)
    matrix_heat_map(P_minus, "P- Predicted Covariance Matrix", " ", " ", col_labels=state_label, row_labels=state_label, savefig_path = fig_path)

    matrix_heat_map(H_design, "H Design Matrix", " ", " ", col_labels=state_label, row_labels=obs_label, savefig_path = fig_path)
    matrix_heat_map(P_plus, "P+ Updated Covariance Matrix", " ", " ", col_labels=state_label, row_labels=state_label, savefig_path = fig_path)

def j2000_to_rtn(r_j2000, v_j2000, error_j2000):
    """
    将 J2000 惯性系下的误差向量转换到 RTN (Radial, Transverse, Normal) 轨道坐标系。
    
    参数:
    r_j2000 : array_like, shape(3,)
        卫星在 J2000 下的位置向量 [x, y, z] (单位: m)
    v_j2000 : array_like, shape(3,)
        卫星在 J2000 下的速度向量 [vx, vy, vz] (单位: m/s)
    error_j2000 : array_like, shape(3,)
        待转换的误差向量 (如: 位置误差 或 速度误差)
        
    返回:
    error_rtn : numpy.ndarray, shape(3,)
        转换后的 RTN 误差向量 [Radial, Transverse, Normal]
    """
    
    # 1. 转换为 numpy 数组以方便计算
    r = np.array(r_j2000, dtype=float)
    v = np.array(v_j2000, dtype=float)
    err = np.array(error_j2000, dtype=float)
    
    # 2. 计算 R (Radial) 轴单位向量
    # R 方向：从地心指向卫星
    r_norm = np.linalg.norm(r)
    if r_norm == 0:
        raise ValueError("位置向量模长为0，无法定义 RTN 坐标系")
    u_r = r / r_norm
    
    # 3. 计算 N (Normal) 轴单位向量
    # N 方向：垂直于轨道平面 (角动量方向 h = r x v)
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    if h_norm == 0:
        raise ValueError("角动量为0 (直线运动)，无法定义 RTN 坐标系")
    u_n = h / h_norm
    
    # 4. 计算 T (Transverse) 轴单位向量
    # T 方向：在轨道平面内，垂直于 R (右手定则: T = N x R)
    u_t = np.cross(u_n, u_r)
    
    # 5. 构建旋转矩阵 (J2000 -> RTN)
    # 因为基向量是正交单位向量，旋转矩阵即为这些基向量行排列
    # M = [u_r^T]
    #     [u_t^T]
    #     [u_n^T]
    rot_matrix = np.array([u_r, u_t, u_n])
    
    # 6. 进行投影 (矩阵乘法)
    error_rtn = rot_matrix @ err
    
    return error_rtn

if __name__ == "__main__":
    starttime = TimeSystem.TimeSystem().from_str('2020-09-01 06:00:00.000') #TimeSystem
    endtime = TimeSystem.TimeSystem().from_str('2020-09-01 09:00:00.000')
    flag = 'Lambda Int Test'
    log_filepath = "D:\\csu\\GNSS_MIX\\CSUAPPS\\CSUPODSApp\\BIN\\LOG\\Diff\\Diff_Log_YYYYMMDD_{}.log".format(flag) 
    if os.path.exists('new'):
        shutil.rmtree('new')
    os.makedirs('new')

    files_path = split_file_by_marker(log_filepath)

    for file_path in files_path:
        print(f"Processing file: {file_path}")
        if 'forward' in file_path:
            mode = 'forward'
        elif 'backward' in file_path:
            mode = 'backward'
        elif 'twoway' in file_path:
            mode = 'twoway'
        elif 'kinematics' in file_path:
            mode = 'kinematics'
            continue
        else:
            mode = ''

        # epochs_data = parse_kalman_log_with_header(file_path, starttime=starttime, endtime=endtime)
        epochs_data = parse_kalman_log_with_header(file_path)
        # visualize_matrix(epochs_data, epoch=epoch)
        fig_path = os.path.join('new', 'FilterVisualizeImgs' + f'_{mode}_' + flag)
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
            print(f"Created directory: {fig_path}")
        if mode == 'twoway':
            parse_two_way_state(epochs_data[3:-1], savefig_path=fig_path)
        elif mode == 'kinematics':
            parse_kin_state(file_path, savefig_path=fig_path)
        else:
            parse_SC_state(epochs_data[3:-1], savefig_path=fig_path)
            visualize_sats(epochs_data[3:-1], savefig_path=fig_path)
    
    '''Plot Residuals in RTN frame (Orbit frame)'''
    log_filepath = "D:\\csu\\GNSS_MIX\\CSUAPPS\\CSUPODSApp\\BIN\\LOG\\Diff\\Diff_Log_YYYYMMDD_{}_twoway.log".format(flag)
    epochs_data = parse_kalman_log_with_header(log_filepath)

    fig_path = os.path.join('new', 'FilterVisualizeImgs_{}_RTN'.format(flag))
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
        print(f"Created directory: {fig_path}")
    time_list = []
    pos_error_rtn_list = []
    vel_error_rtn_list = []
    for epoch_data in epochs_data:
        time_list.append(epoch_data['Timestamp'].datetime)
        Bstate = epoch_data['State Vector']
        BState_res = epoch_data['Smoothed Spacecraft B State Error (GT - Est)']
        r_j2000 = Bstate[0:3]
        v_j2000 = Bstate[3:6]
        pos_error_j2000 = BState_res[0:3]
        vel_error_j2000 = BState_res[3:6]
        pos_error_rtn = j2000_to_rtn(r_j2000, v_j2000, pos_error_j2000)
        vel_error_rtn = j2000_to_rtn(r_j2000, v_j2000, vel_error_j2000)
        pos_error_rtn_list.append(pos_error_rtn)
        vel_error_rtn_list.append(vel_error_rtn)
    plot_state(time_list, np.array(pos_error_rtn_list)*100, ['Radial', 'Transverse', 'Normal'], "Spacecraft B Position Residuals in RTN (GT - Est)", "Position Residuals (cm)", savefig_path=fig_path, print_rmse=True)
    plot_state(time_list, np.array(vel_error_rtn_list)*1000, ['Radial', 'Transverse', 'Normal'], "Spacecraft B Velocity Residuals in RTN (GT - Est)", "Velocity Residuals (mm/s)", savefig_path=fig_path, print_rmse=True)
