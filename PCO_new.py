import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
import json

import TimeSystem
import Sp3Tools
import Rinex
import AttitudeModel
import MathTool

def get_att(att_df : pd.DataFrame, curr_time : TimeSystem.TimeSystem):
    # 使用MathTool中的batch_quaternion_interpolate，
    times_array = np.array(att_df['timesystem'].values)
    time0 = times_array[0]
    times_array = times_array - time0  # 转换为相对时间
    quat_array = np.array(att_df[['q1', 'q2', 'q3', 'q4']].values)  # 转换为 (N, 4) 数组
    target_time = (curr_time - time0)
    interpolated_quat = MathTool.batch_quaternion_interpolate(times_array, quat_array, np.array([target_time]))[0]
    # print(f"Interpolated quaternion at {curr_time}: {interpolated_quat}")
    return AttitudeModel.Attitude(interpolated_quat)

def get_interpolation(sp3_df : pd.DataFrame, curr_time : TimeSystem.TimeSystem, target_sat : str, values=['x', 'y', 'z']):
    if target_sat not in sp3_df['sat_id'].values:
        return [None] * len(values)
    sp3_df = sp3_df[sp3_df['sat_id'] == target_sat]
    times_array = np.array(sp3_df['timesystem'].values)
    time0 = times_array[0]
    times_array = times_array - time0  # 转换为相对时间
    target_time = (curr_time - time0)

    out_y = []
    for val in values:
        if val not in sp3_df.columns:
            raise ValueError(f"SP3 DataFrame does not contain column '{val}'")
        y_array = np.array(sp3_df[val].values)
        y, err = MathTool.window_interpolate(times_array, y_array, target_time)
        out_y.append(y)
    # print(f"Interpolated {values} at {curr_time} for {target_sat}: {out_y}")
    return np.array(out_y)

def compute_corr(currtime : TimeSystem.TimeSystem, leo_sat : np.ndarray, gnss_sat : np.ndarray, att : AttitudeModel.Attitude, pco: np.ndarray):
    # 符号有点乱，能跑就行
    M_J2k_B = att.get_rotation_matrix().T
    M_Ecef_J2k = AttitudeModel.RotM_Ecef_J2k(currtime)
    M_Ecef_B = M_J2k_B @ M_Ecef_J2k.T

    los = gnss_sat - leo_sat
    los_unit = los / np.linalg.norm(los)
    los_ecef = M_Ecef_B @ los_unit

    corr = -np.dot(pco, los_ecef)
    return corr

def apply_pco_correction(corr, obs_values, prn):
    for obs_type, obs_value in obs_values.items():
        if obs_value is None:
            continue
        if obs_type.startswith('C'):
            obs_values[obs_type] -= corr
        elif obs_type.startswith('L'):
            wavelength_m = config['wavelengths'].get(prn[0]+':'+obs_type, None)
            if wavelength_m is not None:
                obs_values[obs_type] -= corr / wavelength_m
        else:
            obs_values[obs_type] = obs_value  # 不修改其他类型的观测值
    return obs_values

if __name__ == "__main__":
    config_path = 'PCOConfig.json'
    config = json.load(open(config_path, 'r'))

    start_time = TimeSystem.TimeSystem()
    start_time = start_time.from_str(config['start_time'])
    end_time = TimeSystem.TimeSystem()
    end_time = end_time.from_str(config['end_time'])
    print(f"Start Time: {start_time}, End Time: {end_time}")

    leo_sp3_df = Sp3Tools.read_sp3(config['leo_sp3_file'], start_time, end_time)
    print(f"LEO SP3 data length: {len(leo_sp3_df)}")
    print(leo_sp3_df.head())

    gnss_sp3_df = Sp3Tools.read_sp3(config['gnss_sp3_file'], start_time, end_time)
    print(f"GNSS SP3 data length: {len(gnss_sp3_df)}")
    print(gnss_sp3_df.head())

    att_df = AttitudeModel.read_attitude_file(config['attitude_file'])
    print(f"Attitude data length: {len(att_df)}")
    print(att_df.head())

    # refTime = TimeSystem.TimeSystem()
    # refTime = refTime.from_str("2024 12 27 04:18:04")

    for antenna in config['antennas']:
        print(f"Processing antenna: {antenna['name']}")
        pco = np.array(antenna['pco'])
        obs_head = Rinex.readObsHead(antenna['input_rinex'])
        obs_data = Rinex.readObs(antenna['input_rinex'])
        print(f"RINEX data length for {antenna['name']}: {len(obs_data)}")
        

        epoch_remove = []
        for epoch, obs in obs_data.items():
            epoch_time = TimeSystem.TimeSystem()
            epoch_time._update_from_datetime(epoch)
            if start_time <= epoch_time <= end_time:
                # if epoch_time == refTime:
                #     print(f"Reference Time {refTime} found in RINEX data.")
                att = get_att(att_df, epoch_time)
                leo_pv = get_interpolation(leo_sp3_df, epoch_time, config['leo_satellite'], ['x', 'y', 'z'])

                for prn, obs_val in obs.items():
                    gnss_pv = get_interpolation(gnss_sp3_df, epoch_time, prn, ['x', 'y', 'z'])
                    if None in gnss_pv:
                        print(f"Warning: No SP3 data for {prn} at {epoch_time}")
                        continue
                    corr = compute_corr(epoch_time, leo_pv, gnss_pv, att, pco)

                    corrected_obs = apply_pco_correction(corr, obs_val, prn)
                    obs_data[epoch][prn] = corrected_obs
            else:
                epoch_remove.append(epoch)

        for epoch in epoch_remove:
            obs_data.pop(epoch)

        Rinex.writeObs(obs_head, obs_data, antenna['output_rinex'])