import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import json
import os 
import sys

sys.path.insert(0, './lib')
import TimeSystem
import Sp3Tools
import Rinex
import AttitudeModel
import MathTool

from astropy.time import Time as astroTime
from astropy.coordinates.builtin_frames.intermediate_rotation_transforms import (
    cirs_to_itrs_mat,
    gcrs_to_cirs_mat,
)

def preprocess_sp3(sp3_df):
    """
    将 SP3 DataFrame 预处理为按 sat_id 索引的 Numpy 字典
    """
    sp3_dict = {}
    if sp3_df.empty:
        return sp3_dict
        
    time0 = sp3_df['timesystem'].iloc[0]
    for sat_id, group in sp3_df.groupby('sat_id'):
        # 预先计算相对时间，避免在循环中重复计算
        times_array = np.array([(ts - time0) for ts in group['timesystem']])
        sp3_dict[sat_id] = {
            'time0': time0,
            'times': times_array,
            'x': group['x'].values,
            'y': group['y'].values,
            'z': group['z'].values
        }
    return sp3_dict

def compute_corr(M_Ecef_J2k, leo_sat, gnss_sat, att_matrix, pco):
    M_Ecef_B = att_matrix @ M_Ecef_J2k.T

    los = gnss_sat - leo_sat
    los_unit = los / np.linalg.norm(los)
    los_ecef = M_Ecef_B @ los_unit

    return -np.dot(pco, los_ecef)

def apply_pco_correction(corr, obs_values, prn, wavelengths):
    for obs_type, obs_value in obs_values.items():
        if obs_value is None:
            continue
        if obs_type.startswith('C'):
            obs_values[obs_type] -= corr
        elif obs_type.startswith('L'):
            wavelength_m = wavelengths.get(prn[0]+':'+obs_type, None)
            if wavelength_m is not None:
                obs_values[obs_type] -= corr / wavelength_m
    return obs_values

def test_mode():
    q = np.array([-0.81572172555333333,0.19423688330666666,0.19392628966666667,-0.50917844349333330])
    att = AttitudeModel.Attitude(q)
    M_J2k_B = att.get_rotation_matrix().T
    time0 = TimeSystem.TimeSystem().from_str("2024-12-27 10:01:04")
    time0 = time0  + (-18)  # 考虑 GPS-UTC 的 18 秒差异
    t = astroTime(time0.to_str(), scale='utc')
    R_gcrs_to_cirs = gcrs_to_cirs_mat(t)
    R_cirs_to_itrs = cirs_to_itrs_mat(t)
    M_Ecef_J2k = R_cirs_to_itrs @ R_gcrs_to_cirs

    # M_Ecef_J2k = AttitudeModel.RotM_Ecef_J2k(time0)
    # M_Ecef_J2k = np.array([[-0.40267797861927479, -0.91534115289702933, 0.0010096277690443949],
    #                         [0.91533848744602953, -0.40267924284775958, -0.0022092486437789781],
    #                         [0.0024287723462297173, 3.4535376856215478e-05, 0.99999704993174743]])

    M_Ecef_B = M_J2k_B @ M_Ecef_J2k
    print("M_J2k_B:\n", M_J2k_B)
    print("M_Ecef_J2k:\n", M_Ecef_J2k)
    print("M_Ecef_B:\n", M_Ecef_B)

    gnss_sat = np.array([-16340732.025444839, -11087479.525186287, 19793755.230494007])
    leo_sat = np.array([2283180.6620000000,1173269.0860000001, 6340205.3619999997])
    pco = np.array([-0.7664, -0.1690, 0.1579])
    corr = compute_corr(M_Ecef_J2k, leo_sat, gnss_sat, M_J2k_B, pco)
    print("Computed correction:", corr)
    

if __name__ == "__main__":
    # test_mode()

    config_path = './configs/PCOConfig1227.json'
    config = json.load(open(config_path, 'r'))

    start_time = TimeSystem.TimeSystem().from_str(config['start_time'])+(-300)
    end_time = TimeSystem.TimeSystem().from_str(config['end_time'])+300

    # 正确代码：直接传入 config 读取到的 list
    print("Loading SP3 and Attitude Data...")
    leo_sp3_df = Sp3Tools.read_sp3(config['leo_sp3_file'], start_time, end_time)
    gnss_sp3_df = Sp3Tools.read_sp3(config['gnss_sp3_file'], start_time, end_time)
    att_df = AttitudeModel.read_attitude_file(config['attitude_file'])

    # --- 1. 核心优化：预处理 SP3 数据结构 ---
    leo_sp3_dict = preprocess_sp3(leo_sp3_df)
    gnss_sp3_dict = preprocess_sp3(gnss_sp3_df)

    # --- 2. 核心优化：姿态预准备 ---
    att_time0 = att_df['timesystem'].iloc[0]
    att_times_array = np.array([(ts - att_time0) for ts in att_df['timesystem']])
    att_quat_array = att_df[['q1', 'q2', 'q3', 'q4']].values

    for antenna in config['antennas']:
        print(f"Processing antenna: {antenna['name']}")
        pco = np.array(antenna['pco'])
        obs_head = Rinex.readObsHead(antenna['input_rinex'])
        obs_data = Rinex.readObs(antenna['input_rinex'])
        
        epoch_remove = []
        valid_epochs = []
        valid_target_times = []
        
        # 第一次遍历：筛选有效历元，收集插值所需的时间点
        for epoch in obs_data.keys():
            epoch_ts = TimeSystem.TimeSystem()
            epoch_ts._update_from_datetime(epoch)
            if start_time <= epoch_ts <= end_time:
                valid_epochs.append((epoch, epoch_ts))
                valid_target_times.append(epoch_ts - att_time0)
            else:
                epoch_remove.append(epoch)

        for epoch in epoch_remove:
            obs_data.pop(epoch)

        # --- 3. 核心优化：批量执行姿态插值 ---
        print("Batch interpolating attitudes...")
        if valid_target_times:
            # ATT-L1B may start a few seconds after the first observation and
            # end before 24:00.  Hold the nearest endpoint attitude rather
            # than failing Slerp for those boundary epochs.
            target_times = np.asarray(valid_target_times, dtype=float)
            target_times = np.clip(target_times, att_times_array[0], att_times_array[-1])
            interpolated_quats = MathTool.batch_quaternion_interpolate(
                att_times_array, att_quat_array, target_times
            )
        
        # 第二次遍历：执行修正
        print("Applying corrections...")
        corr_ls = []
        for idx, (epoch_dt, epoch_ts) in enumerate(valid_epochs):
            # 获取当前历元的预结算姿态矩阵和地球自转矩阵
            att_matrix = AttitudeModel.Attitude(interpolated_quats[idx]).get_rotation_matrix().T
            # M_Ecef_J2k = AttitudeModel.RotM_Ecef_J2k(epoch_ts)

            t0 = epoch_ts + (-18)
            t = astroTime(t0.to_str(), scale='utc')
            R_gcrs_to_cirs = gcrs_to_cirs_mat(t)
            R_cirs_to_itrs = cirs_to_itrs_mat(t)
            M_Ecef_J2k = R_cirs_to_itrs @ R_gcrs_to_cirs
            
            leo_sat_id = config['leo_satellite']
            if leo_sat_id not in leo_sp3_dict:
                continue
                
            # LEO 卫星位置插值 (避免了 DataFrame 切片)
            leo_data = leo_sp3_dict[leo_sat_id]
            target_time_leo = epoch_ts - leo_data['time0']
            if target_time_leo < leo_data['times'][0] or target_time_leo > leo_data['times'][-1]:
                # The single-day reference orbit can cover only a partial
                # tracking arc; leave observations outside that arc unchanged.
                continue
            leo_pv = np.array([
                MathTool.window_interpolate(leo_data['times'], leo_data['x'], target_time_leo)[0],
                MathTool.window_interpolate(leo_data['times'], leo_data['y'], target_time_leo)[0],
                MathTool.window_interpolate(leo_data['times'], leo_data['z'], target_time_leo)[0]
            ])

            for prn, obs_val in obs_data[epoch_dt].items():
                if prn not in gnss_sp3_dict:
                    continue
                    
                gnss_data = gnss_sp3_dict[prn]
                target_time_gnss = epoch_ts - gnss_data['time0']
                
                # 检查是否越界 (MathTool 会报错，这里预先规避或者捕获)
                if target_time_gnss < gnss_data['times'][0] or target_time_gnss > gnss_data['times'][-1]:
                    continue

                gnss_pv = np.array([
                    MathTool.window_interpolate(gnss_data['times'], gnss_data['x'], target_time_gnss)[0],
                    MathTool.window_interpolate(gnss_data['times'], gnss_data['y'], target_time_gnss)[0],
                    MathTool.window_interpolate(gnss_data['times'], gnss_data['z'], target_time_gnss)[0]
                ])

                corr = compute_corr(M_Ecef_J2k, leo_pv, gnss_pv, att_matrix, pco)
                corr_ls.append(corr)
                corrected_obs = apply_pco_correction(corr, obs_val, prn, config.get('wavelengths', {}))
                obs_data[epoch_dt][prn] = corrected_obs

        Rinex.writeObs(obs_head, obs_data, antenna['output_rinex'])
        print(f"corr_mean: {np.mean(corr_ls):.6f}, corr_std: {np.std(corr_ls):.6f}, corr_max: {np.max(corr_ls):.6f}, corr_min: {np.min(corr_ls):.6f}")
