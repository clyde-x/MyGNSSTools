import sys
import os
sys.path.append("../lib")

import numpy as np
import pandas as pd

import TimeSystem
import Rinex
import Sp3Tools
import SatID
import Sp3Tools
import AntPCO
from Coord import GNSSCoordinate as GC
import MathTool
import json


import parse_rtk_stat
import parse_rtk_pos

from KF import RelativeClockKF

config_name = "config.json"


curr_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(curr_dir, config_name)
with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

stat_file = os.path.join(curr_dir, cfg["file_path"]["stat_file"])
# pos_file = os.path.join(curr_dir, cfg["file_path"]["pos_file"])
gnss_file = os.path.join(curr_dir, cfg["file_path"]["gnss_sp3"])
clk_file = os.path.join(curr_dir, cfg["file_path"]["clk_file"])
obsA_file = os.path.join(curr_dir, cfg["file_path"]["obsA"])
obsB_file = os.path.join(curr_dir, cfg["file_path"]["obsB"])
igs_file = os.path.join(curr_dir, cfg["file_path"]["atx_igs"])
igs_parser = AntPCO.ATXParserRTK(igs_file)
sys_list = cfg["system_list"]
obs_types = cfg.get("observations", ["PC", "LC"])
starttime = TimeSystem.TimeSystem().from_str(cfg["time_range"]["start"])
endtime = TimeSystem.TimeSystem().from_str(cfg["time_range"]["end"])


# read stat and pos files
stat = parse_rtk_stat.parse_file(stat_file)
# outpos_df, pos_comment = parse_rtk_pos.parse_file(pos_file)

# read sp3, rinex obs and atx files
gnss_sp3, gnss_eph = Sp3Tools.read_sp3([gnss_file], returnsp3=True)
gnss_eph = gnss_eph[gnss_eph['sat_id'].str[0].isin(sys_list)]
all_sats = gnss_sp3.sat_id_list
all_sats = SatID.filter(all_sats, sys_list=sys_list)

obsA_head = Rinex.readObsHead(obsA_file)
obsB_head = Rinex.readObsHead(obsB_file)
obsA = Rinex.readObs(obsA_file, obsA_head)
obsB = Rinex.readObs(obsB_file, obsB_head)
time_list_A = obsA.keys()
time_list_B = obsB.keys()
time_list = sorted(set(time_list_A).intersection(set(time_list_B)))
time_list = [TimeSystem.TimeSystem().from_datetime(t) for t in time_list if starttime.datetime <= t <= endtime.datetime]

if cfg["refA"]["mode"] == "static":
    posA = np.array(cfg["refA"]["pos"])
    ecefA = GC.pos2ecef(posA, 'deg')
    ecefA_dict ={}
    for t in time_list:
        ecefA_dict[t] = ecefA
elif cfg["refA"]["mode"] == "moving":
    pos_file = os.path.join(curr_dir, cfg["refA"]["pos_file"])
    outpos_df, pos_comment = parse_rtk_pos.parse_file(pos_file)
    ecefA_dict = {}
    ecefA_list = []
    for i, t in enumerate(outpos_df['timesystem'].values):
        pos = outpos_df.loc[i, ['latitude(deg)', 'longitude(deg)', 'height(m)']].values
        ecef = GC.pos2ecef(pos, 'deg')
        ecefA_list.append(ecef)
    ecefA_list = np.array(ecefA_list)
    time_array = outpos_df['timesystem'].values
    for t in time_list:
        # ecef = MathTool.slow_motion_interpolate(t, time_array, ecefA_list)
        ecef, err = MathTool.window_interpolate(time_array, ecefA_list, t)
        ecefA_dict[t] = ecef

if cfg["refB"]["mode"] == "static":
    posB = np.array(cfg["refB"]["pos"])
    ecefB = GC.pos2ecef(posB, 'deg')
    ecefB_dict = {}
    for t in time_list:
        ecefB_dict[t] = ecefB
elif cfg["refB"]["mode"] == "moving":
    pos_file = os.path.join(curr_dir, cfg["refB"]["pos_file"])
    outpos_df, pos_comment = parse_rtk_pos.parse_file(pos_file)
    data_ = outpos_df[['latitude(deg)', 'longitude(deg)', 'height(m)']].values
    ecefB_list = []
    ecefB_dict = {}
    for i, t in enumerate(outpos_df['timesystem'].values):
        ecef = GC.pos2ecef(data_[i], 'deg')
        ecefB_list.append(ecef)
    ecefB_list = np.array(ecefB_list)
    time_array = outpos_df['timesystem'].values
    for t in time_list:
        # ecef = MathTool.slow_motion_interpolate(t, time_array, ecefB_list)
        ecef, err = MathTool.window_interpolate(time_array, ecefB_list, t)
        # print(f"Interpolated position for {t}: {ecef}")
        ecefB_dict[t] = ecef

CONST_C = 299792458.0

obsA_df = Rinex.reduce_rinex(obsA, sys_list=sys_list, starttime=starttime.datetime, endtime=endtime.datetime, eph_sats=all_sats, exclude_sats=cfg["exclude_sats"])
obsB_df = Rinex.reduce_rinex(obsB, sys_list=sys_list, starttime=starttime.datetime, endtime=endtime.datetime, eph_sats=all_sats, exclude_sats=cfg["exclude_sats"])

sat_freq_set = set()
freq_set = set()
for obs in obsA_df.itertuples():
    tupleA = (obs.sat_id, obs.atx_freq1)
    tupleB = (obs.sat_id, obs.atx_freq2)
    sat_freq_set.add(tupleA)
    sat_freq_set.add(tupleB)
    freq_set.update((obs.atx_freq1,obs.atx_freq2))

sat_pcv = {}
for prn, freq in sat_freq_set:
    tmp_sat_pcv = igs_parser.get_pcv(prn)
    if not tmp_sat_pcv:
        print(f"在 ATX 文件中未找到卫星 {prn} 的 PCO/PCV 数据！")
        continue
    if prn not in sat_pcv.keys():
        tmp_sat_pco1 = tmp_sat_pcv.off.get(freq)
        tmp_sat_noazi1 = tmp_sat_pcv.var.get(freq)
        if tmp_sat_pco1 is None or tmp_sat_noazi1 is None:
            print(f"在 ATX 文件中未找到卫星 {prn} 的频率 {freq} 的 PCO/PCV 数据！")
            continue
        
        sat_pcv[prn] = {
            'pcvt': tmp_sat_pcv,
            'data' : {freq: {'pco': tmp_sat_pco1, 'noazi': tmp_sat_noazi1}}
        }
    else:
        tmp_sat_pco1 = tmp_sat_pcv.off.get(freq)
        tmp_sat_noazi1 = tmp_sat_pcv.var.get(freq)
        if tmp_sat_pco1 is None or tmp_sat_noazi1 is None:
            print(f"在 ATX 文件中未找到卫星 {prn} 的频率 {freq} 的 PCO/PCV 数据！")
            continue
        sat_pcv[prn]['data'][freq] = {'pco': tmp_sat_pco1, 'noazi': tmp_sat_noazi1}
print("A Antenna Type:", obsA_head['Antenna Type'], "B Antenna Type:", obsB_head['Antenna Type'])
print("A Antenna Delta:", obsA_head['Antenna Delta'], "B Antenna Delta:", obsB_head['Antenna Delta'])

obsA_ant = igs_parser.get_receiver_pcv(obsA_head['Antenna Type']) # type: PCV_t
obsB_ant = igs_parser.get_receiver_pcv(obsB_head['Antenna Type'])

obsA_pcv = {}
obsB_pcv = {}

if obsA_ant is not None and obsB_ant is not None:
    print("在 ATX 文件中未找到接收机天线的 PCO/PCV 数据！")
    for freq in freq_set:
        pcoA_enu = obsA_ant.off.get(freq)
        pcoB_enu = obsB_ant.off.get(freq)
        if pcoA_enu is None or pcoB_enu is None:
            pcoA_enu = [0.0, 0.0, 0.0]
            pcoB_enu = [0.0, 0.0, 0.0]
            print(f"在 ATX 文件中未找到接收机天线的 {freq} 频率的 PCO 数据！")
        pcoA_noazi = obsA_ant.var.get(freq)
        pcoB_noazi = obsB_ant.var.get(freq)
        if pcoA_noazi is None or pcoB_noazi is None:
            pcoA_noazi = [0.0]
            pcoB_noazi = [0.0]
            print(f"在 ATX 文件中未找到接收机天线的 {freq} 频率的无方位角 PCV 数据！")
        obsA_pcv[freq] = {'pco': pcoA_enu, 'noazi': pcoA_noazi}
        obsB_pcv[freq] = {'pco': pcoB_enu, 'noazi': pcoB_noazi}

# ================================================================
#  build stat lookup dict for KF
#  key: (time_str, sat_id_str) -> {'L1': SatState, 'L2': SatState}
# ================================================================
stat_dict = {}
for s in stat:
    if s.type == 'SAT':
        key = (s.timesystem.to_str(), str(s.sat_id))
        if key not in stat_dict:
            stat_dict[key] = {}
        stat_dict[key][f'L{s.frq}'] = s

print(f"stat_dict: {len(stat_dict)} satellite-epoch entries loaded")

# ================================================================
#  TropState lookup: parse $TROP records for ZTD
#  key: (time_str, rcv) -> ztd [meters]
# ================================================================
trop_dict = {}
for s in stat:
    if s.type == 'TROP':
        time_str = s.timesystem.to_str()
        if time_str not in trop_dict:
            trop_dict[time_str] = {}
        trop_dict[time_str][s.rcv] = s.ztd  # 1=rover, 2=base
print(f"trop_dict: {len(trop_dict)} ZTD epochs loaded")

# simple mapping function for slant troposphere
def tropo_map(el_deg):
    """Simple continued-fraction mapping function."""
    el_rad = max(np.radians(el_deg), 0.01)
    return 1.0 / (np.sin(el_rad) + 0.00143 / (np.tan(el_rad) + 0.0445))

# ================================================================
#  initialize Kalman filter (clock + N_IF in state)
# ================================================================
kf = RelativeClockKF(
    tau=1e6,                # near random-walk clock
    sigma_clock=0.3,         # clock process noise [m]
    sigma_N=1e-8,            # N process noise [m]
    sigma_N_init=1.0,        # N init variance [m²]
    sigma_P=2.0,             # IF pseudorange obs noise [m]
    sigma_L=0.02,            # IF carrier phase obs noise [m]
)
kf.enable_logging()         # enable diagnostic recording

### KF 钟差估计

clock_results = []
prev_time = None
SNR_THRESHOLD = 25  # dbHz
EL_CUTOFF = 15.0     # elevation cutoff [deg]

# ---- data editing and ambiguity feedback buffers ----
MW_WINDOW = 5                       # sliding window for code-phase divergence
mw_history = {}                      # {sat_id: list of MW_linear values}
nw_history = {}                      # {sat_id: list of N_wl values} for MW wide-lane QC


for t in time_list:
    ecefA = ecefA_dict.get(t)
    ecefB = ecefB_dict.get(t)
    t_str = t.to_str()
    t_tai_str = (t + 19).to_str()  # convert GPS time to TAI for Coord functions

    curr_obsA = obsA_df[obsA_df['timesystem'] == t]
    curr_obsB = obsB_df[obsB_df['timesystem'] == t]

    common_sats = set(curr_obsA['sat_id'].values).intersection(set(curr_obsB['sat_id'].values))
    common_sats = SatID.sort(common_sats, ifstr=True)

    if len(common_sats) == 0:
        continue

    # sun ECEF position (same for all satellites in this epoch)
    sun_ecef = GC.get_sun_ecef_astropy(t_tai_str)

    observations = []
    elevations = {}

    for sat_id in common_sats:
        if sat_id in cfg["exclude_sats"]:
            continue
        # ---- stat lookup ----
        stat_key = (t_str, sat_id)
        sat_stats = stat_dict.get(stat_key, {})

        # cycle slip detection
        has_slip = False
        for freq_label in ('L1', 'L2'):
            ss = sat_stats.get(freq_label)
            if ss and ss.slip != 0:
                has_slip = True
                break
        if has_slip:
            kf.reset_sat(sat_id)
            mw_history.pop(sat_id, None)   # clear MW history on cycle slip

        # SNR quality check
        snr_ok = True
        for freq_label in ('L1', 'L2'):
            ss = sat_stats.get(freq_label)
            if ss and ss.snr < SNR_THRESHOLD:
                snr_ok = False
                break
        if not snr_ok:
            continue

        # elevation angle and azimuth (default if not in stat)
        el_deg = 45.0
        az_deg = 0.0
        s_l1 = sat_stats.get('L1')
        if s_l1 is not None:
            el_deg = s_l1.el
            az_deg = s_l1.az
        elevations[sat_id] = np.radians(el_deg)

        # elevation cutoff
        if el_deg < EL_CUTOFF:
            continue

        # RTKLIB residual QC: check resp (pseudorange) and resc (carrier phase)
        resid_ok = True
        for freq_label in ('L1', 'L2'):
            ss = sat_stats.get(freq_label)
            if ss:
                if ss.vsat == 0:            # RTKLIB marked as invalid
                    resid_ok = False; break
                if abs(ss.resp) > 5.0:      # pseudorange residual > 5m
                    resid_ok = False; break
                if abs(ss.resc) > 0.1:      # carrier residual > 10cm
                    resid_ok = False; break
        if not resid_ok:
            continue

        # ---- geometric range via SP3 interpolation ----
        sat_pos, err = Sp3Tools.gnss_eph_interpolate(gnss_eph, sat_id, t)
        if sat_pos is None or err is None:
            print(f"Warning: Invalid satellite position for {sat_id} at {t_str}, skipping.")
            continue
        sat_ecef = np.array([sat_pos[0], sat_pos[1], sat_pos[2]]) * 1000
        rhoA = np.linalg.norm(ecefA - sat_ecef)
        rhoB = np.linalg.norm(ecefB - sat_ecef)
        rho_AB = rhoB - rhoA

        # ---- raw observations ----
        obsA_sat = curr_obsA[curr_obsA['sat_id'] == sat_id]
        obsB_sat = curr_obsB[curr_obsB['sat_id'] == sat_id]
        if len(obsA_sat) == 0 or len(obsB_sat) == 0:
            continue

        C1_A = obsA_sat['P1'].values[0]
        C2_A = obsA_sat['P2'].values[0]
        L1_A = obsA_sat['L1'].values[0]  # cycles
        L2_A = obsA_sat['L2'].values[0]  # cycles

        C1_B = obsB_sat['P1'].values[0]
        C2_B = obsB_sat['P2'].values[0]
        L1_B = obsB_sat['L1'].values[0]  # cycles
        L2_B = obsB_sat['L2'].values[0]  # cycles

        Alambda1 = CONST_C/obsA_sat['f1'].values[0]
        Alambda2 = CONST_C/obsA_sat['f2'].values[0]
        Blambda1 = CONST_C/obsB_sat['f1'].values[0]
        Blambda2 = CONST_C/obsB_sat['f2'].values[0]

        # ---- PCO/PCV corrections (satellite + receiver antenna phase center) ----
        atx_freqA1 = obsA_sat['atx_freq1'].values[0]
        atx_freqA2 = obsA_sat['atx_freq2'].values[0]
        atx_freqB1 = obsB_sat['atx_freq1'].values[0]
        atx_freqB2 = obsB_sat['atx_freq2'].values[0]
        tmp_pcvt = sat_pcv.get(SatID.SatID(sat_id)).get('pcvt')
        tmp_pcv = sat_pcv.get(SatID.SatID(sat_id)).get('data')
        # print(tmp_pcvt, tmp_pcv, atx_freqA1, atx_freqA2)
        # print(obsA_pcv.keys())
        try:
            if tmp_pcv is not None and sat_id[0] in ('G', 'C', 'R', 'E'):
                corrA_l1 = AntPCO.correct_sat_rec_ant(
                    0, sat_ecef, ecefA, sun_ecef, atx_freqA1,
                    tmp_pcvt, tmp_pcv, obsA_ant, obsA_pcv, el_deg, az_deg, obsA_head['Antenna Delta'])
                corrB_l1 = AntPCO.correct_sat_rec_ant(
                    0, sat_ecef, ecefB, sun_ecef, atx_freqB1,
                    tmp_pcvt, tmp_pcv, obsB_ant, obsB_pcv, el_deg, az_deg, obsB_head['Antenna Delta'])
                corrA_l2 = AntPCO.correct_sat_rec_ant(
                    0, sat_ecef, ecefA, sun_ecef, atx_freqA2,
                    tmp_pcvt, tmp_pcv, obsA_ant, obsA_pcv, el_deg, az_deg, obsA_head['Antenna Delta'])
                corrB_l2 = AntPCO.correct_sat_rec_ant(
                    0, sat_ecef, ecefB, sun_ecef, atx_freqB2,
                    tmp_pcvt, tmp_pcv, obsB_ant, obsB_pcv, el_deg, az_deg, obsB_head['Antenna Delta'])

            else:
                corrA_l1 = corrB_l1 = corrA_l2 = corrB_l2 = 0.0
        except Exception as e:
            print(f"PCO/PCV correction error for {sat_id} at {t_str}: {e}")
            corrA_l1 = corrB_l1 = corrA_l2 = corrB_l2 = 0.0

        # ---- troposphere correction (SD, from RTKLIB ZTD) ----
        tropoA = 0.0; tropoB = 0.0
        if t_str in trop_dict:
            t_dict = trop_dict[t_str]
            ztdA = t_dict.get(1, 0.0)   # rcv=1 rover
            ztdB = t_dict.get(2, 0.0)   # rcv=2 base
            m_el = tropo_map(el_deg)
            tropoA = ztdA * m_el
            tropoB = ztdB * m_el
        SD_tropo = tropoB - tropoA

        # ---- IF combination SD observations ----
        # P_IF_A = alpha*(C1_A+corrA_l1) + beta*(C2_A+corrA_l2)
        alpha_if = Alambda1**2 / (Alambda1**2 - Alambda2**2)
        beta_if = -Alambda2**2 / (Alambda1**2 - Alambda2**2)
        alpha_mw = Alambda1 / (Alambda1 - Alambda2)
        beta_mw = -Alambda2 / (Alambda1 - Alambda2)
        gamma_mw_p = Alambda1 / (Alambda1 + Alambda2)
        delta_mw_p = Alambda2 / (Alambda1 + Alambda2)
        lambda_wl = CONST_C / (Alambda1 - Alambda2)
        P_IF_A = alpha_if * (C1_A + corrA_l1) + beta_if * (C2_A + corrA_l2)
        P_IF_B = alpha_if * (C1_B + corrB_l1) + beta_if * (C2_B + corrB_l2)
        SD_P_IF = P_IF_B - P_IF_A - rho_AB - SD_tropo

        # L_IF_A = alpha*(L1_A*lambda1+corrA_l1) + beta*(L2_A*lambda2+corrA_l2)
        L_IF_A = alpha_if * (L1_A * Alambda1 + corrA_l1) + beta_if * (L2_A * Alambda2 + corrA_l2)
        L_IF_B = alpha_if * (L1_B * Blambda1 + corrB_l1) + beta_if * (L2_B * Blambda2 + corrB_l2)
        SD_L_IF = L_IF_B - L_IF_A - rho_AB - SD_tropo

        # ---- MW wide-lane QC / N_w estimation ----
        # MW = (f1*L1 - f2*L2)/(f1-f2) - (f1*P1 + f2*P2)/(f1+f2)
        SD_L1 = (L1_B - L1_A) * Alambda1
        SD_L2 = (L2_B - L2_A) * Alambda2
        SD_P1 = C1_B - C1_A
        SD_P2 = C2_B - C2_A
        MW_sd = (alpha_mw * SD_L1 + beta_mw * SD_L2) - (gamma_mw_p * SD_P1 + delta_mw_p * SD_P2)
        N_wl = MW_sd / lambda_wl  # wide-lane ambiguity estimate [cycles]
        if sat_id in nw_history and len(nw_history[sat_id]) >= 3:
            nw_arr = np.array(nw_history[sat_id])
            nw_mean = np.mean(nw_arr)
            nw_std = np.std(nw_arr) + 1e-8
            if abs(N_wl - nw_mean) > 3.0 * nw_std:
                # wide-lane jump: reset KF and clear history
                kf.reset_sat(sat_id)
                nw_history[sat_id] = []
        # always update history (even after reset, new arc starts)
        if sat_id not in nw_history:
            nw_history[sat_id] = []
        nw_history[sat_id].append(N_wl)
        if len(nw_history[sat_id]) > MW_WINDOW:
            nw_history[sat_id].pop(0)

        # ---- code-phase divergence check (MW linear combination) ----
        # MW = (L1_B - L1_A)*λ1 - (C1_B - C1_A) ≈ λ1*N1 [constant within arc]
        MW_raw = (L1_B - L1_A) * Alambda1 - (C1_B - C1_A)
        if sat_id not in mw_history:
            mw_history[sat_id] = []
        mw_history[sat_id].append(MW_raw)
        if len(mw_history[sat_id]) > MW_WINDOW:
            mw_history[sat_id].pop(0)
        if len(mw_history[sat_id]) >= 3:
            mw_arr = np.array(mw_history[sat_id])
            mw_mean = np.mean(mw_arr)
            mw_std = np.std(mw_arr) + 1e-8
            if abs(MW_raw - mw_mean) > 3.0 * mw_std:
                continue  # reject satellite: code-phase divergence detected

        # ---- raw SD observations (with PCO/PCV and tropo corrections) ----
        SD_P1_raw = (C1_B + corrB_l1) - (C1_A + corrA_l1) - rho_AB - SD_tropo
        SD_P2_raw = (C2_B + corrB_l2) - (C2_A + corrA_l2) - rho_AB - SD_tropo
        SD_L1_raw = (L1_B * Blambda1 + corrB_l1) - (L1_A * Alambda1 + corrA_l1) - rho_AB - SD_tropo
        SD_L2_raw = (L2_B * Blambda2 + corrB_l2) - (L2_A * Alambda2 + corrA_l2) - rho_AB - SD_tropo

        # ---- build observation dict based on config ----
        obs_to_check = []
        if "PC" in obs_types:
            obs_to_check.append(SD_P_IF)
        if "LC" in obs_types:
            obs_to_check.append(SD_L_IF)

        if obs_to_check:
            if np.any(np.isnan(obs_to_check)) or np.any(np.isinf(obs_to_check)):
                continue

        obs_entry = {'sat': sat_id}
        use_P1 = "P1" in obs_types or "C1" in obs_types
        use_P2 = "P2" in obs_types or "C2" in obs_types
        use_L1 = "L1" in obs_types
        use_L2 = "L2" in obs_types
        use_PC = "PC" in obs_types
        use_LC = "LC" in obs_types

        if use_PC:
            obs_entry['PC'] = SD_P_IF
        if use_LC:
            obs_entry['LC'] = SD_L_IF
        if use_P1:
            obs_entry['P1'] = SD_P1_raw
        if use_P2:
            obs_entry['P2'] = SD_P2_raw
        if use_L1:
            obs_entry['L1'] = SD_L1_raw
        if use_L2:
            obs_entry['L2'] = SD_L2_raw

        observations.append(obs_entry)

    if not observations:
        continue

    # ---- KF predict + update ----
    if prev_time is not None:
        dt = t - prev_time  # seconds
        kf.predict(dt)
    else:
        # first epoch: initialize clock from pseudorange median
        if use_PC:
            p_vals = [o['PC'] for o in observations]
        elif use_P1:
            p_vals = [o['P1'] for o in observations]
        else:
            p_vals = [obs_entry.get('PC', obs_entry.get('P1', 0.0)) for obs_entry in observations]
        kf.x[0] = np.median(p_vals)
        kf.P[0, 0] = 0.01  # small, trust pseudorange

    result = kf.update(observations, elevations)
    prev_time = t

    if result is None:
        # NaN prevented — skip recording this epoch
        continue

    # attach time to the latest KF diagnostic log entry
    if kf._log_entries:
        kf._log_entries[-1]['time'] = TimeSystem.TimeSystem().from_str(t_str).datetime

    # ---- clean up MW history for satellites no longer observed ----
    for sat_id in list(mw_history.keys()):
        if sat_id not in [o['sat'] for o in observations]:
            del mw_history[sat_id]
    for sat_id in list(nw_history.keys()):
        if sat_id not in [o['sat'] for o in observations]:
            del nw_history[sat_id]

    clock_results.append({
        'time': TimeSystem.TimeSystem().from_str(t_str).datetime,
        'clock_m': kf.clock,
        'N': kf.ambiguity()
    })

    if len(clock_results) % 100 == 0:
        print(f"KF 进度: {len(clock_results)} 历元, 当前钟差: {kf.clock:.4f} m, {kf.clock*10/3} ns")

# ================================================================
#  save clock estimates
# ================================================================
clock_df = pd.DataFrame(clock_results)
clock_file = os.path.join(curr_dir, cfg["output"]["clockcsv"])
clock_df.to_csv(clock_file, index=False)
print(f"钟差估计完成, 共 {len(clock_results)} 个历元, 结果保存至 {clock_file}")

# ================================================================
#  save KF diagnostic log (lightweight summary only)
# ================================================================
kf_log = kf.get_log()
if kf_log:
    summary_rows = []
    for entry in kf_log:
        summary_rows.append({
            'time': entry.get('time'),
            'dt': entry['dt'],
            'n_sats': entry['n_sats'],
            'n_state': entry['n_state'],
            'x_pred_clock_m': entry['x_pred_clock'],
            'x_upd_clock_m': entry['x_upd_clock'],
            'P_pred_clock': entry['P_pred_clock'],
            'P_upd_clock': entry['P_upd_clock'],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_file = os.path.join(curr_dir, cfg["output"]["summary"])
    summary_df.to_csv(summary_file, index=False)
    print(f"KF诊断摘要: {len(kf_log)} 个历元, 保存至 {summary_file}")

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(clock_df['time'], clock_df['clock_m']*1e9/CONST_C, label='KF Clock Estimate (ns)')
std = np.std(clock_df['clock_m']*1e9/CONST_C)
plt.text(0.5, 0.9, f'Standard Deviation: {std:.2f} ns', transform=plt.gca().transAxes, fontsize=12, ha='center')
print(f"钟差估计标准差: {std:.2f} ns")
plt.xlabel('Time')
plt.ylabel('Clock Difference (ns)')
plt.title('Estimated Clock Difference Over Time')
plt.xticks(rotation=45)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(curr_dir, cfg["output"]["clockpng"]), dpi=300)