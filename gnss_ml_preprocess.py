# -*- coding: utf-8 -*-
"""
GNSS观测数据AI预处理原型
=========================
从RINEX观测文件和SP3精密轨道文件提取特征，
为周跳探测分类和粗差异常检测训练Random Forest基础模型。

用法:
    python gnss_ml_preprocess.py

输出:
    D:/csu/GNSS_MIX/ml_output/
        features_A.csv, features_B.csv    -- 特征数据集
        cycle_slip_report.txt             -- 周跳分类器评估报告
        outlier_report.txt                -- 粗差检测器评估报告
        feature_importance_cs.png         -- 周跳特征重要性
        feature_importance_outlier.png    -- 粗差特征重要性
        timeline_predictions.png          -- 时间序列预测可视化
"""

import os
import sys
import re
import warnings
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -- 添加 MyTools 到路径 --
MYTOOLS_DIR = r'D:\csu\MyTools'
if MYTOOLS_DIR not in sys.path:
    sys.path.insert(0, MYTOOLS_DIR)

from Rinex import readObs, readObsHead
from Sp3Tools import SP3

# scikit-learn
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_recall_curve,
                             f1_score, accuracy_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib

warnings.filterwarnings('ignore')

# =============================================================================
# 常量
# =============================================================================
C_LIGHT = 299792458.0                     # m/s
F1_GPS  = 1575.42e6                       # Hz
F2_GPS  = 1227.60e6                       # Hz
LAMBDA1 = C_LIGHT / F1_GPS               # m  (~0.1903m)
LAMBDA2 = C_LIGHT / F2_GPS               # m  (~0.2442m)
F1_SQ   = F1_GPS ** 2
F2_SQ   = F2_GPS ** 2

# 数据文件路径
BIN_DIR = r'D:\csu\GNSS_MIX\CSUAPPS\CSUPODSApp\BIN'
RINEX_A = os.path.join(BIN_DIR, 'inputdata', 'Obs', 'Diff', 'GFCN2450.20O')
RINEX_B = os.path.join(BIN_DIR, 'inputdata', 'Obs', 'Diff', 'GFDN2450.20O')
SP3_A   = os.path.join(BIN_DIR, 'inputdata', 'Obs', 'Diff', 'GFO_C_smooth_kbr.sp3')
SP3_B   = os.path.join(BIN_DIR, 'inputdata', 'Obs', 'Diff', 'GFO_D_smooth_kbr.sp3')
LOG_FILE = os.path.join(BIN_DIR, 'LOG', 'Diff',
                        'Diff_Log_YYYYMMDD_dataedit_test_cs.log')

OUTPUT_DIR = r'D:\csu\GNSS_MIX\ml_output'

# =============================================================================
# 特征提取
# =============================================================================
class GNSSFeatureExtractor:
    """从RINEX+SP3提取每颗卫星每历元的特征向量"""

    # RINEX 2.x -> 3.x 映射后的观测类型键名 (GPS)
    # L1->L1C, L2->L2P, C1->C1C, P1->C1P, P2->C2P, S1->S1C, S2->S2P
    OBS_L1 = 'L1C'
    OBS_L2 = 'L2P'
    OBS_C1 = 'C1C'
    OBS_P1 = 'C1P'
    OBS_P2 = 'C2P'
    OBS_S1 = 'S1C'
    OBS_S2 = 'S2C'

    def __init__(self, rinex_file, sp3_file, label='A'):
        self.rinex_file = rinex_file
        self.sp3_file = sp3_file
        self.label = label
        self.obsHead = None
        self.obsData = None
        self.sp3 = None
        self.sp3_positions = {}  # {datetime: np.array([x,y,z])}

    def load_data(self):
        """加载RINEX和SP3"""
        print(f"[{self.label}] 加载RINEX: {os.path.basename(self.rinex_file)}")
        self.obsHead = readObsHead(self.rinex_file, needSatList=True)
        self.obsData = readObs(self.rinex_file, self.obsHead)

        if self.obsData is None:
            raise RuntimeError(f"无法读取RINEX文件: {self.rinex_file}")

        print(f"  历元数: {len(self.obsData)}")
        print(f"  卫星: {self.obsHead['prn']}")
        print(f"  观测类型: {self.obsHead['OBS TYPES']}")

        print(f"[{self.label}] 加载SP3: {os.path.basename(self.sp3_file)}")
        self.sp3 = SP3(self.sp3_file)
        # 建立 datetime -> position 查找表
        # SP3类解析结果在 sp3.data (list of dicts)，df可能为空
        for epoch_entry in self.sp3.data:
            t = epoch_entry.get('time', None)
            if t is None:
                continue
            # TimeSystem -> datetime conversion
            if hasattr(t, 'to_datetime'):
                dt = t.to_datetime()
            elif hasattr(t, 'datetime'):
                dt = t.datetime
            else:
                dt = t
            sats = epoch_entry.get('satellites', {})
            # LEO SP3通常只有一颗卫星 (L01)
            for sat_id, sat_data in sats.items():
                pos_tuple = sat_data.get('position', (0, 0, 0))
                pos = np.array(pos_tuple, dtype=float)
                # SP3坐标单位是km，转换为m
                if abs(pos[0]) < 1e5:
                    pos *= 1000.0
                self.sp3_positions[dt] = pos
                break  # 只取第一颗卫星
        print(f"  SP3历元数: {len(self.sp3_positions)}")

    def _get_obs(self, epoch, prn, obs_type):
        """安全获取观测值，返回None如果不存在"""
        try:
            val = self.obsData[epoch][prn].get(obs_type, None)
            if val is not None and val != 0.0:
                return float(val)
        except (KeyError, TypeError):
            pass
        return None

    def _get_leo_pos(self, epoch):
        """获取LEO卫星在给定历元的位置(米)"""
        if epoch in self.sp3_positions:
            return self.sp3_positions[epoch]
        # 找最近的SP3历元 (容差5秒)
        min_dt = None
        min_diff = 999999
        for sp3_t in self.sp3_positions:
            diff = abs((sp3_t - epoch).total_seconds())
            if diff < min_diff:
                min_diff = diff
                min_dt = sp3_t
        if min_dt is not None and min_diff <= 5.0:
            return self.sp3_positions[min_dt]
        return None

    def extract_features(self):
        """
        逐历元逐卫星提取全部特征，返回DataFrame
        每行 = 一个 (epoch, prn) 对
        """
        epochs = sorted(self.obsData.keys())
        records = []

        # 上一历元数据缓存 (per-satellite)
        prev_data = {}  # prn -> {LI, MW, PC, LC, ...}
        # MW滑动统计 (per-satellite arc)
        mw_accum = {}   # prn -> {'sum': float, 'sum_sq': float, 'n': int}

        for ei, epoch in enumerate(epochs):
            prns_in_epoch = list(self.obsData[epoch].keys())

            for prn in prns_in_epoch:
                if not prn.startswith('G'):
                    continue  # 只处理GPS

                # --- 原始观测值 ---
                L1_cyc = self._get_obs(epoch, prn, self.OBS_L1)
                L2_cyc = self._get_obs(epoch, prn, self.OBS_L2)
                P1 = self._get_obs(epoch, prn, self.OBS_P1)
                P2 = self._get_obs(epoch, prn, self.OBS_P2)
                C1 = self._get_obs(epoch, prn, self.OBS_C1)
                S1 = self._get_obs(epoch, prn, self.OBS_S1)
                S2 = self._get_obs(epoch, prn, self.OBS_S2)

                # 码观测: 优先P1，否则C1
                code1 = P1 if P1 is not None else C1
                code2 = P2

                # 完整性检查
                if L1_cyc is None or L2_cyc is None or code1 is None or code2 is None:
                    continue

                # --- 相位转为米 ---
                L1_m = L1_cyc * LAMBDA1
                L2_m = L2_cyc * LAMBDA2

                # --- 线性组合 ---
                # LI = L1 - L2 (几何无关组合，单位:米)
                LI = L1_m - L2_m

                # MW (Melbourne-Wubbena) 组合 (单位:周)
                # MW = (f1*L1 - f2*L2)/(f1-f2) - (f1*P1 + f2*P2)/(f1+f2)
                # L1_cyc, L2_cyc 是cycles
                phase_WL = (F1_GPS * L1_cyc - F2_GPS * L2_cyc) / (F1_GPS - F2_GPS)
                code_NL  = (F1_GPS * code1 + F2_GPS * code2) / (F1_GPS + F2_GPS)
                lambda_WL = C_LIGHT / (F1_GPS - F2_GPS)
                MW = phase_WL - code_NL / lambda_WL  # 宽巷周数

                # PC = 无电离层码组合
                PC = (F1_SQ * code1 - F2_SQ * code2) / (F1_SQ - F2_SQ)

                # LC = 无电离层相位组合
                LC = (F1_SQ * L1_m - F2_SQ * L2_m) / (F1_SQ - F2_SQ)

                # P1 - L1 (多径+模糊度指标)
                P1_minus_L1 = code1 - L1_m

                # GF (Geometry-Free) 码组合: P2 - P1
                GF_code = code2 - code1

                # --- 历元间差分特征 ---
                is_new_sat = (prn not in prev_data)

                dLI = np.nan
                dMW = np.nan
                dPC = np.nan
                dLC = np.nan
                div = np.nan
                MW_mean = np.nan
                MW_std = np.nan

                if not is_new_sat:
                    pd_ = prev_data[prn]
                    dLI = LI - pd_['LI']
                    dMW = MW - pd_['MW']
                    dPC = PC - pd_['PC']
                    dLC = LC - pd_['LC']
                    div = dPC - dLC  # 码相发散

                    # MW 滑动统计
                    if prn in mw_accum:
                        ma = mw_accum[prn]
                        ma['n'] += 1
                        ma['sum'] += MW
                        ma['sum_sq'] += MW * MW
                        MW_mean = ma['sum'] / ma['n']
                        if ma['n'] > 1:
                            MW_std = np.sqrt(ma['sum_sq'] / ma['n'] - MW_mean ** 2)
                            MW_std = max(MW_std, 0.0)
                        else:
                            MW_std = 0.3  # 初始值
                    else:
                        mw_accum[prn] = {'sum': MW, 'sum_sq': MW * MW, 'n': 1}
                        MW_mean = MW
                        MW_std = 0.3
                else:
                    # 新星: 重置MW累积
                    mw_accum[prn] = {'sum': MW, 'sum_sq': MW * MW, 'n': 1}
                    MW_mean = MW
                    MW_std = 0.3

                # --- 几何距离 (近似) ---
                # 用SP3给出的LEO位置估算到GPS卫星的"伪距"
                # code_res = PC - 估计的rho (粗略，含钟差)
                leo_pos = self._get_leo_pos(epoch)
                code_res = np.nan
                if leo_pos is not None:
                    # PC本身就约等于 rho + 钟差，所以PC在不同卫星间的波动反映几何差异
                    # 我们不需要真正的rho，只要 PC 的卫星间一致性
                    code_res = PC  # 后面通过星间单差消除钟差

                # --- 构造记录 ---
                rec = {
                    'epoch': epoch,
                    'prn': prn,
                    'epoch_idx': ei,
                    # 原始观测
                    'S1': S1 if S1 is not None else np.nan,
                    'S2': S2 if S2 is not None else np.nan,
                    # 组合量
                    'LI': LI,
                    'MW': MW,
                    'PC': PC,
                    'LC': LC,
                    'P1_minus_L1': P1_minus_L1,
                    'GF_code': GF_code,
                    # 历元间差分
                    'dLI': dLI,
                    'dMW': dMW,
                    'dPC': dPC,
                    'dLC': dLC,
                    'div': div,
                    # 滑动统计
                    'MW_mean': MW_mean,
                    'MW_std': MW_std,
                    # MW标准化偏差 (类TurboEdit)
                    'MW_norm': (MW - MW_mean) / MW_std if MW_std > 1e-10 else 0.0,
                    # 码残差 (用于星间单差)
                    'code_res': code_res,
                    # 标志
                    'is_new_sat': 1 if is_new_sat else 0,
                }
                records.append(rec)

                # 更新上一历元缓存
                prev_data[prn] = {
                    'LI': LI, 'MW': MW, 'PC': PC, 'LC': LC,
                    'L1_m': L1_m, 'L2_m': L2_m,
                }

            # 清除不再出现的卫星的prev_data (检测NewSat)
            current_gps = {p for p in prns_in_epoch if p.startswith('G')}
            lost_sats = set(prev_data.keys()) - current_gps
            for sat in lost_sats:
                del prev_data[sat]
                if sat in mw_accum:
                    del mw_accum[sat]

        df = pd.DataFrame(records)
        print(f"[{self.label}] 特征提取完成: {len(df)} 条记录, "
              f"{df['prn'].nunique()} 颗卫星, {df['epoch_idx'].nunique()} 个历元")
        return df


def compute_single_difference_features(df_A, df_B):
    """
    计算星间单差特征 (A-B)
    合并两星在相同(epoch, prn)的特征
    """
    # 按 epoch + prn 合并
    merged = pd.merge(df_A, df_B, on=['epoch', 'prn', 'epoch_idx'],
                      suffixes=('_A', '_B'), how='inner')

    # 星间单差码残差 (消除接收机钟差)
    merged['code_SD_res'] = merged['code_res_A'] - merged['code_res_B']

    # 星间单差相位组合
    merged['LC_SD'] = merged['LC_A'] - merged['LC_B']
    merged['PC_SD'] = merged['PC_A'] - merged['PC_B']

    # 星间单差dLI
    merged['dLI_SD'] = merged['dLI_A'] - merged['dLI_B']

    # 逐历元计算code_SD_res的中位数和MAD (用于粗差标签)
    epoch_stats = merged.groupby('epoch')['code_SD_res'].agg(
        median='median',
        mad=lambda x: np.median(np.abs(x - np.median(x))) * 1.4826
    ).reset_index()
    epoch_stats.columns = ['epoch', 'code_SD_median', 'code_SD_mad']
    merged = merged.merge(epoch_stats, on='epoch', how='left')

    # 标准化码单差残差
    merged['code_SD_norm'] = np.where(
        merged['code_SD_mad'] > 1e-6,
        (merged['code_SD_res'] - merged['code_SD_median']) / merged['code_SD_mad'],
        0.0
    )

    print(f"星间单差合并完成: {len(merged)} 条记录")
    return merged


# =============================================================================
# C++ 日志解析器
# =============================================================================
class DiffLogParser:
    """
    解析 DiffDataEditor 的C++日志，提取每个 (epoch, prn) 的数据编辑标签。
    
    周跳标签: 0=Normal, 1=IonSlip, 2=MWSlip, 3=DivSlip, 4=NewSat
    粗差标签: 0=Normal, 1=Outlier
    """

    # 正则表达式 (预编译)
    RE_EPOCH = re.compile(
        r'^--- Epoch #(\d+): (\d{4})/(\d{2})/(\d{2}) (\d+):(\d+):(\d+\.?\d*) ---')
    RE_NEWSAT = re.compile(
        r'\[NewSat\] GPS (\d+)')
    RE_CS_ION = re.compile(
        r'\[CycleSlip-Ion\] GPS (\d+), dIon=([\d.eE+-]+)')
    RE_CS_DIV = re.compile(
        r'\[CycleSlip-Div\] GPS (\d+), divA=([\d.eE+-]+), divB=([\d.eE+-]+)')
    RE_CS_MW = re.compile(
        r'\[CycleSlip-MW\] GPS (\d+), MWA=([\d.eE+-]+), MWB=([\d.eE+-]+)')
    RE_CODE_REJECT = re.compile(
        r'\[CodeGross\] Round \d+: REJECT GPS (\d+)')
    RE_PHASE_SLIP = re.compile(
        r'\[PhaseSlip\] Round \d+: SLIP GPS (\d+)')
    RE_RAIM_CODE = re.compile(
        r'\[RAIM-Code\] REJECT GPS (\d+)')
    RE_RAIM_PHASE = re.compile(
        r'\[RAIM-Phase\] MARK SLIP GPS (\d+)')
    RE_BLACKLIST_SKIP = re.compile(
        r'\[Blacklist\] SKIP GPS (\d+)')
    RE_BLACKLIST_TRIGGER = re.compile(
        r'\[Blacklist\] TRIGGERED: GPS (\d+)')
    RE_EPOCH_REJECT = re.compile(
        r'\*\* EPOCH REJECTED')
    RE_COMMON_SATS = re.compile(
        r'After RemoveNonCommon: CommonSats=(\d+)')

    def __init__(self, log_file):
        self.log_file = log_file
        # {(datetime, 'G04'): {'cs_label': int, 'out_label': int,
        #                       'cs_detail': str, 'out_detail': str}}
        self.labels = {}
        # {datetime: set of 'G04' prns} — 每个历元的共视卫星列表
        self.epoch_sats = {}
        # 所有非拒绝的日志历元时间戳 (用于时间对齐)
        self.all_epochs = []
        self.n_epochs = 0
        self.stats = {'NewSat': 0, 'IonSlip': 0, 'MWSlip': 0, 'DivSlip': 0,
                      'CodeGross': 0, 'PhaseSlip': 0, 'RAIM_Code': 0,
                      'RAIM_Phase': 0, 'Blacklist': 0, 'EpochReject': 0}

    @staticmethod
    def _gps_prn(num_str):
        """'4' -> 'G04'"""
        return f'G{int(num_str):02d}'

    def _set_cs_label(self, epoch_dt, prn, label, detail):
        """设置周跳标签 (不覆盖更高优先级)"""
        key = (epoch_dt, prn)
        if key not in self.labels:
            self.labels[key] = {'cs_label': 0, 'out_label': 0,
                                'cs_detail': '', 'out_detail': ''}
        cur = self.labels[key]['cs_label']
        # 优先级: NewSat(4) > IonSlip(1) > MWSlip(2) > DivSlip(3)
        # 如果已有更高优先级标签，不覆盖
        priority = {0: 0, 4: 4, 1: 3, 2: 2, 3: 1}
        if priority.get(label, 0) > priority.get(cur, 0):
            self.labels[key]['cs_label'] = label
            self.labels[key]['cs_detail'] = detail

    def _set_out_label(self, epoch_dt, prn, detail):
        """设置粗差标签 (二分类: 1=Outlier)"""
        key = (epoch_dt, prn)
        if key not in self.labels:
            self.labels[key] = {'cs_label': 0, 'out_label': 0,
                                'cs_detail': '', 'out_detail': ''}
        self.labels[key]['out_label'] = 1
        self.labels[key]['out_detail'] = detail

    def parse(self):
        """解析日志文件，返回标签字典"""
        print(f"解析C++日志: {os.path.basename(self.log_file)}")

        cur_epoch_dt = None
        cur_epoch_rejected = False
        cur_sats = set()

        with open(self.log_file, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.rstrip()

                # --- 历元头 ---
                m = self.RE_EPOCH.match(line)
                if m:
                    # 保存上一历元的共视卫星
                    if cur_epoch_dt is not None and cur_sats:
                        self.epoch_sats[cur_epoch_dt] = cur_sats
                    self.n_epochs += 1
                    yr, mo, dy = int(m.group(2)), int(m.group(3)), int(m.group(4))
                    hr, mn = int(m.group(5)), int(m.group(6))
                    sec_f = float(m.group(7))
                    sec = int(sec_f)
                    usec = int((sec_f - sec) * 1e6)
                    cur_epoch_dt = datetime.datetime(yr, mo, dy, hr, mn, sec, usec)
                    cur_epoch_rejected = False
                    cur_sats = set()
                    self.all_epochs.append(cur_epoch_dt)
                    continue

                if cur_epoch_dt is None:
                    continue

                # --- 整历元拒绝 ---
                if self.RE_EPOCH_REJECT.search(line):
                    cur_epoch_rejected = True
                    self.stats['EpochReject'] += 1
                    continue

                if cur_epoch_rejected:
                    continue

                # --- 共视卫星 (跟踪列表) ---
                if 'SatA: ' in line or 'SatB: ' in line:
                    # 从SatA/SatB行提取PRN
                    continue
                m_common = self.RE_COMMON_SATS.search(line)
                if m_common:
                    continue

                # --- 周跳标记 ---
                m = self.RE_NEWSAT.search(line)
                if m:
                    prn = self._gps_prn(m.group(1))
                    self._set_cs_label(cur_epoch_dt, prn, 4, 'NewSat')
                    cur_sats.add(prn)
                    self.stats['NewSat'] += 1
                    continue

                m = self.RE_CS_ION.search(line)
                if m:
                    prn = self._gps_prn(m.group(1))
                    self._set_cs_label(cur_epoch_dt, prn, 1,
                                       f'Ion dIon={m.group(2)}')
                    self.stats['IonSlip'] += 1
                    continue

                m = self.RE_CS_DIV.search(line)
                if m:
                    prn = self._gps_prn(m.group(1))
                    self._set_cs_label(cur_epoch_dt, prn, 3,
                                       f'Div A={m.group(2)} B={m.group(3)}')
                    self.stats['DivSlip'] += 1
                    continue

                m = self.RE_CS_MW.search(line)
                if m:
                    prn = self._gps_prn(m.group(1))
                    self._set_cs_label(cur_epoch_dt, prn, 2,
                                       f'MW A={m.group(2)} B={m.group(3)}')
                    self.stats['MWSlip'] += 1
                    continue

                # --- 粗差标记 ---
                m = self.RE_CODE_REJECT.search(line)
                if m:
                    prn = self._gps_prn(m.group(1))
                    self._set_out_label(cur_epoch_dt, prn, 'CodeGross')
                    self.stats['CodeGross'] += 1
                    continue

                m = self.RE_PHASE_SLIP.search(line)
                if m:
                    prn = self._gps_prn(m.group(1))
                    # PhaseSlip 同时标记为周跳和粗差
                    self._set_cs_label(cur_epoch_dt, prn, 3, 'PhaseSlip-TD')
                    self._set_out_label(cur_epoch_dt, prn, 'PhaseSlip')
                    self.stats['PhaseSlip'] += 1
                    continue

                m = self.RE_RAIM_CODE.search(line)
                if m:
                    prn = self._gps_prn(m.group(1))
                    self._set_out_label(cur_epoch_dt, prn, 'RAIM_Code')
                    self.stats['RAIM_Code'] += 1
                    continue

                m = self.RE_RAIM_PHASE.search(line)
                if m:
                    prn = self._gps_prn(m.group(1))
                    self._set_cs_label(cur_epoch_dt, prn, 3, 'RAIM_Phase')
                    self.stats['RAIM_Phase'] += 1
                    continue

                m = self.RE_BLACKLIST_SKIP.search(line)
                if m:
                    prn = self._gps_prn(m.group(1))
                    self._set_out_label(cur_epoch_dt, prn, 'Blacklist')
                    self.stats['Blacklist'] += 1
                    continue

                m = self.RE_BLACKLIST_TRIGGER.search(line)
                if m:
                    prn = self._gps_prn(m.group(1))
                    self._set_out_label(cur_epoch_dt, prn, 'Blacklist')
                    self.stats['Blacklist'] += 1
                    continue

                # 收集共视卫星 (从GPS PRN标签行)
                if line.strip().startswith('GPS '):
                    for tok in line.split('\t'):
                        tok = tok.strip()
                        if tok.startswith('GPS '):
                            try:
                                cur_sats.add(self._gps_prn(tok.split()[1]))
                            except (IndexError, ValueError):
                                pass

        # 保存最后一个历元的卫星
        if cur_epoch_dt is not None and cur_sats:
            self.epoch_sats[cur_epoch_dt] = cur_sats

        print(f"  日志历元数: {self.n_epochs}")
        print(f"  标签事件统计:")
        for k, v in self.stats.items():
            if v > 0:
                print(f"    {k}: {v}")
        print(f"  有标签的(epoch,prn)对: {len(self.labels)}")

        return self.labels


def merge_log_labels(df, log_parser, tolerance_sec=3.0):
    """
    将日志解析的标签合并到特征DataFrame。
    自动检测并补偿日志时间(GPST)与RINEX时间(UTC)之间的系统偏移。
    
    df: 特征DataFrame，需要有 'epoch' 和 'prn' 列
    log_parser: DiffLogParser 实例 (已调用 parse())
    tolerance_sec: 时间匹配容差（秒）
    
    返回: 新增 cs_label_log, out_label_log 列的 DataFrame
    """
    log_labels = log_parser.labels
    # 使用全部日志历元做时间对齐 (包括无事件的正常历元)
    log_epochs = sorted(set(log_parser.all_epochs))
    
    cs_labels = np.zeros(len(df), dtype=int)
    out_labels = np.zeros(len(df), dtype=int)
    matched_count = 0
    event_count = 0
    
    feature_epochs = sorted(df['epoch'].unique())
    # 统一转为 pd.Timestamp 再取 .timestamp()，避免 Python datetime 与
    # pandas Timestamp 对 naive datetime 时区解释不一致 (UTC vs 本地时间)
    log_epoch_arr = np.array([pd.Timestamp(e).timestamp() for e in log_epochs])
    feat_epoch_arr = np.array([pd.Timestamp(e).timestamp() for e in feature_epochs])
    
    # 自动检测时间偏移 (GPST - UTC)
    # 扫描整数偏移，用严格容差(0.5s)确定最佳偏移
    best_offset = 0
    best_match_count = 0
    for test_offset in range(-5, 25):
        shifted = log_epoch_arr - test_offset
        cnt = 0
        for ft in feat_epoch_arr[:500]:
            if np.min(np.abs(shifted - ft)) <= 0.5:
                cnt += 1
        if cnt > best_match_count:
            best_match_count = cnt
            best_offset = test_offset
    
    print(f"  时间偏移检测: 最佳offset={best_offset}s "
          f"(前500历元中{best_match_count}个匹配, 日志=GPST, RINEX=UTC)")
    
    # 预建查找表: 将特征epoch映射到最近的日志epoch
    epoch_map = {}  # feature_epoch -> log_epoch
    shifted_log = log_epoch_arr - best_offset  # 将日志时间转为RINEX时间域
    for fe in feature_epochs:
        fe_ts = fe.timestamp()
        diffs = np.abs(shifted_log - fe_ts)
        min_idx = np.argmin(diffs)
        if diffs[min_idx] <= tolerance_sec:
            epoch_map[fe] = log_epochs[min_idx]
    
    print(f"  时间对齐: {len(epoch_map)}/{len(feature_epochs)} 个特征历元匹配到日志")
    
    for i, row in enumerate(df.itertuples()):
        epoch = row.epoch
        prn = row.prn
        
        log_epoch = epoch_map.get(epoch)
        if log_epoch is None:
            continue
        
        matched_count += 1
        key = (log_epoch, prn)
        if key in log_labels:
            cs_labels[i] = log_labels[key]['cs_label']
            out_labels[i] = log_labels[key]['out_label']
            if cs_labels[i] != 0 or out_labels[i] != 0:
                event_count += 1
    
    df = df.copy()
    df['cs_label_log'] = cs_labels
    df['out_label_log'] = out_labels
    
    print(f"  标签合并完成: {matched_count} 行匹配, {event_count} 个非Normal事件")
    
    return df


# =============================================================================
# 标签生成 (阈值规则 — 作为对照)
# =============================================================================
def generate_cycle_slip_labels(df):
    """
    基于观测特征自动生成周跳标签 (5类)
    0 = Normal, 1 = IonSlip, 2 = MWSlip, 3 = DivSlip, 4 = NewSat
    """
    labels = np.zeros(len(df), dtype=int)

    # NewSat 标记 (使用A星的标志，A/B应一致)
    new_sat_col = 'is_new_sat_A' if 'is_new_sat_A' in df.columns else 'is_new_sat'
    labels[df[new_sat_col] == 1] = 4

    # 周跳检测 (仅对非NewSat的记录)
    mask_not_new = labels == 0

    # 电离层跳变: |dLI_A| > 0.2m 或 |dLI_B| > 0.2m
    ION_TH = 0.2  # m
    if 'dLI_A' in df.columns:
        ion_slip = mask_not_new & (
            (df['dLI_A'].abs() > ION_TH) | (df['dLI_B'].abs() > ION_TH)
        )
    else:
        ion_slip = mask_not_new & (df['dLI'].abs() > ION_TH)
    labels[ion_slip] = 1

    # MW跳变: |dMW| > 4*MW_std (TurboEdit标准) 且 MW_std > 0.05
    MW_K = 4.0
    if 'MW_norm_A' in df.columns:
        mw_slip = mask_not_new & (labels == 0) & (
            (df['MW_norm_A'].abs() > MW_K) | (df['MW_norm_B'].abs() > MW_K)
        )
    else:
        mw_slip = mask_not_new & (labels == 0) & (df['MW_norm'].abs() > MW_K)
    labels[mw_slip] = 2

    # 码相发散: |div| > 3m
    DIV_TH = 3.0  # m
    if 'div_A' in df.columns:
        div_slip = mask_not_new & (labels == 0) & (
            (df['div_A'].abs() > DIV_TH) | (df['div_B'].abs() > DIV_TH)
        )
    else:
        div_slip = mask_not_new & (labels == 0) & (df['div'].abs() > DIV_TH)
    labels[div_slip] = 3

    return labels


def generate_outlier_labels(df):
    """
    基于MAD的鲁棒方法生成粗差标签
    0 = Normal, 1 = Outlier
    """
    labels = np.zeros(len(df), dtype=int)

    # 码单差残差超出 3*MAD
    OUTLIER_K = 3.0
    outlier = df['code_SD_norm'].abs() > OUTLIER_K
    labels[outlier] = 1

    return labels


# =============================================================================
# 模型训练与评估
# =============================================================================
class CycleSlipClassifier:
    """周跳分类器 (Random Forest 多分类)"""

    CLASS_NAMES = ['Normal', 'IonSlip', 'MWSlip', 'DivSlip', 'NewSat']

    def __init__(self, n_estimators=200, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_leaf=5,
            class_weight='balanced',  # 自动处理类别不平衡
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_names = None

    def train(self, X_train, y_train, feature_names=None):
        self.feature_names = feature_names
        print(f"\n{'='*60}")
        print("训练周跳分类器 (Random Forest)")
        print(f"  训练集: {len(X_train)} 样本")
        unique, counts = np.unique(y_train, return_counts=True)
        for u, c in zip(unique, counts):
            name = self.CLASS_NAMES[u] if u < len(self.CLASS_NAMES) else f'Class{u}'
            print(f"    {name}: {c} ({c/len(y_train)*100:.1f}%)")
        self.model.fit(X_train, y_train)
        print("  训练完成")

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        print(f"\n--- 周跳分类器评估 (测试集: {len(X_test)} 样本) ---")

        # 只报告出现在数据中的类别
        present_labels = sorted(set(y_test) | set(y_pred))
        target_names = [self.CLASS_NAMES[i] if i < len(self.CLASS_NAMES)
                        else f'Class{i}' for i in present_labels]

        report = classification_report(y_test, y_pred,
                                       labels=present_labels,
                                       target_names=target_names)
        print(report)

        cm = confusion_matrix(y_test, y_pred, labels=present_labels)
        print("混淆矩阵:")
        print(cm)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"\n准确率: {acc:.4f}, 加权F1: {f1:.4f}")

        return report, cm

    def feature_importance(self, top_n=15):
        importances = self.model.feature_importances_
        if self.feature_names is not None:
            indices = np.argsort(importances)[::-1][:top_n]
            print(f"\n--- 周跳分类 Top-{top_n} 特征重要性 ---")
            for rank, idx in enumerate(indices):
                print(f"  {rank+1:2d}. {self.feature_names[idx]:20s}: {importances[idx]:.4f}")
            return [(self.feature_names[i], importances[i]) for i in indices]
        return importances


class TwoStageCycleSlipClassifier:
    """
    二阶段周跳分类器 (SMOTE + 阈值调优)
    Stage-1: 二分类 (Normal=0 vs AnySlip=1)
        - SMOTE 过采样平衡训练集
        - predict_proba + 自动阈值选择 (最大化 F-beta, beta=2 偏重 recall)
    Stage-2: 4分类 (IonSlip=1, MWSlip=2, DivSlip=3, NewSat=4)
        - SMOTE 过采样
        - 仅对 Stage-1 判定为周跳的样本细分
    """

    CLASS_NAMES = ['Normal', 'IonSlip', 'MWSlip', 'DivSlip', 'NewSat']
    SLIP_NAMES = ['IonSlip', 'MWSlip', 'DivSlip', 'NewSat']

    def __init__(self, n_estimators=300, random_state=42):
        self.stage1 = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=18,
            min_samples_leaf=3,
            class_weight='balanced_subsample',
            random_state=random_state,
            n_jobs=-1
        )
        self.stage2 = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=12,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_names = None
        self.threshold = 0.5  # Stage-1 默认阈值，训练时自动调优

    def train(self, X_train, y_train, feature_names=None):
        self.feature_names = feature_names

        # --- Stage-1: Normal(0) vs AnySlip(1) + SMOTE ---
        y_binary = (y_train > 0).astype(int)
        print(f"\n{'='*60}")
        print("训练二阶段周跳分类器 (SMOTE + 自适应阈值)")
        print(f"{'='*60}")
        print(f"\n--- Stage-1: 二分类 (Normal vs AnySlip) ---")
        print(f"  原始训练集: {len(X_train)} 样本")
        for lbl, name in [(0, 'Normal'), (1, 'AnySlip')]:
            cnt = (y_binary == lbl).sum()
            print(f"    {name}: {cnt} ({cnt/len(y_binary)*100:.1f}%)")

        # SMOTE 过采样
        smote = SMOTE(random_state=42, k_neighbors=min(5, (y_binary == 1).sum() - 1))
        X_s1, y_s1 = smote.fit_resample(X_train, y_binary)
        print(f"  SMOTE 后: {len(X_s1)} 样本")
        for lbl, name in [(0, 'Normal'), (1, 'AnySlip')]:
            cnt = (y_s1 == lbl).sum()
            print(f"    {name}: {cnt} ({cnt/len(y_s1)*100:.1f}%)")

        self.stage1.fit(X_s1, y_s1)
        print("  Stage-1 训练完成")

        # 自适应阈值: 在原始分布上用交叉验证选择最优阈值
        self._tune_threshold_cv(X_train, y_binary)

        # --- Stage-2: 4分类 (仅周跳样本) + SMOTE ---
        slip_mask = y_train > 0
        X_slip = X_train[slip_mask]
        y_slip = y_train[slip_mask]
        print(f"\n--- Stage-2: 4分类 (周跳类型细分) ---")
        print(f"  原始训练集: {len(X_slip)} 样本")
        unique, counts = np.unique(y_slip, return_counts=True)
        for u, c in zip(unique, counts):
            name = self.CLASS_NAMES[u] if u < len(self.CLASS_NAMES) else f'Class{u}'
            print(f"    {name}: {c} ({c/len(y_slip)*100:.1f}%)")

        # Stage-2 SMOTE (少数类可能只有几个样本, 需要调小k_neighbors)
        min_class_count = min(counts)
        if min_class_count >= 2:
            k_s2 = min(5, min_class_count - 1)
            smote2 = SMOTE(random_state=42, k_neighbors=k_s2)
            X_s2, y_s2 = smote2.fit_resample(X_slip, y_slip)
            print(f"  SMOTE 后: {len(X_s2)} 样本")
            unique2, counts2 = np.unique(y_s2, return_counts=True)
            for u, c in zip(unique2, counts2):
                name = self.CLASS_NAMES[u] if u < len(self.CLASS_NAMES) else f'Class{u}'
                print(f"    {name}: {c}")
        else:
            print(f"  最小类仅 {min_class_count} 样本，跳过 SMOTE")
            X_s2, y_s2 = X_slip, y_slip

        self.stage2.fit(X_s2, y_s2)
        print("  Stage-2 训练完成")

    def _tune_threshold_cv(self, X_orig, y_binary_orig):
        """
        自适应阈值选择: 在原始分布上用 StratifiedKFold 交叉验证。
        
        关键思路:
        1. 将原始训练集分 3 折，每折用 SMOTE 训练临时 RF
        2. 在 out-of-fold 数据（原始分布）上收集 predict_proba
        3. 在真实分布的 OOF 概率上搜索 F-beta(beta=2) 最优阈值
        
        这样阈值天然匹配真实数据分布，无需人工缩放系数。
        """
        print("\n  自适应阈值选择 (3-fold CV on original distribution)...")
        
        n_folds = 3
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # 收集所有 OOF 概率
        oof_proba = np.zeros(len(X_orig))
        
        for fold_i, (train_idx, val_idx) in enumerate(skf.split(X_orig, y_binary_orig)):
            X_tr, y_tr = X_orig[train_idx], y_binary_orig[train_idx]
            X_val = X_orig[val_idx]
            
            # SMOTE on this fold's training data
            n_pos = (y_tr == 1).sum()
            if n_pos < 2:
                # 极端情况: 正类不足，跳过 SMOTE
                X_sm, y_sm = X_tr, y_tr
            else:
                sm = SMOTE(random_state=42, k_neighbors=min(5, n_pos - 1))
                X_sm, y_sm = sm.fit_resample(X_tr, y_tr)
            
            # 训练临时 RF (参数与 self.stage1 一致，但不需要 OOB)
            tmp_rf = RandomForestClassifier(
                n_estimators=self.stage1.n_estimators,
                max_depth=self.stage1.max_depth,
                min_samples_leaf=self.stage1.min_samples_leaf,
                class_weight='balanced_subsample',
                random_state=42 + fold_i,
                n_jobs=-1
            )
            tmp_rf.fit(X_sm, y_sm)
            
            # 在验证集（原始分布）上收集概率
            oof_proba[val_idx] = tmp_rf.predict_proba(X_val)[:, 1]
        
        # 在全部 OOF 概率上搜索 F2 最优阈值
        precisions, recalls, thresholds = precision_recall_curve(
            y_binary_orig, oof_proba)
        
        beta = 2.0
        best_fb = 0.0
        best_thr = 0.1
        for i in range(len(thresholds)):
            p, r = precisions[i], recalls[i]
            fb = (1 + beta**2) * p * r / (beta**2 * p + r) if (beta**2 * p + r) > 0 else 0
            if fb > best_fb:
                best_fb = fb
                best_thr = thresholds[i]
        
        self.threshold = best_thr
        
        # 报告 OOF 上的性能
        y_oof = (oof_proba >= self.threshold).astype(int)
        tp = ((y_oof == 1) & (y_binary_orig == 1)).sum()
        fp = ((y_oof == 1) & (y_binary_orig == 0)).sum()
        fn = ((y_oof == 0) & (y_binary_orig == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f2 = (1 + beta**2) * prec * rec / (beta**2 * prec + rec) if (beta**2 * prec + rec) > 0 else 0
        
        auc_pr = np.trapz(precisions, recalls)
        
        print(f"  自适应阈值 = {self.threshold:.4f} (F2最优)")
        print(f"    OOF — Precision={prec:.4f}, Recall={rec:.4f}, F2={f2:.4f}")
        print(f"    OOF — TP={tp}, FP={fp}, FN={fn}, AUC-PR={auc_pr:.4f}")
        
        # 展示几个参考阈值点
        print(f"    {'Threshold':>10s}  {'Precision':>9s}  {'Recall':>6s}  {'F2':>6s}")
        for thr_val in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, self.threshold]:
            yp = (oof_proba >= thr_val).astype(int)
            t_tp = ((yp == 1) & (y_binary_orig == 1)).sum()
            t_fp = ((yp == 1) & (y_binary_orig == 0)).sum()
            t_fn = ((yp == 0) & (y_binary_orig == 1)).sum()
            t_p = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 0
            t_r = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0
            t_f2 = (1 + beta**2) * t_p * t_r / (beta**2 * t_p + t_r) if (beta**2 * t_p + t_r) > 0 else 0
            marker = " <-- 选定" if abs(thr_val - self.threshold) < 1e-6 else ""
            print(f"    {thr_val:10.4f}  {t_p:9.4f}  {t_r:6.4f}  {t_f2:6.4f}{marker}")

    def predict(self, X):
        """二阶段预测: Stage-1 概率阈值 → Stage-2 细分"""
        y_final = np.zeros(len(X), dtype=int)
        # Stage-1: 用调优后的阈值
        proba = self.stage1.predict_proba(X)[:, 1]
        slip_mask = proba >= self.threshold
        # Stage-2: 仅对预测为周跳的样本细分
        if slip_mask.any():
            y_final[slip_mask] = self.stage2.predict(X[slip_mask])
        return y_final

    def evaluate(self, X_test, y_test):
        """分阶段评估 + 整体评估"""
        y_pred = self.predict(X_test)

        # --- Stage-1 评估: 二分类 ---
        y_true_binary = (y_test > 0).astype(int)
        y_pred_binary = (y_pred > 0).astype(int)
        print(f"\n--- Stage-1 评估: 二分类 (threshold={self.threshold:.4f}) ---")
        s1_report = classification_report(
            y_true_binary, y_pred_binary,
            target_names=['Normal', 'AnySlip'])
        print(s1_report)

        # Stage-1 PR 曲线关键点
        proba = self.stage1.predict_proba(X_test)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(
            y_true_binary, proba)
        print(f"  Stage-1 AUC-PR: {np.trapz(precisions, recalls):.4f}")
        # 展示几个关键阈值点
        print(f"  {'Threshold':>10s}  {'Precision':>9s}  {'Recall':>6s}  {'F1':>6s}")
        for thr_val in [0.1, 0.2, 0.3, 0.4, 0.5, self.threshold]:
            yp = (proba >= thr_val).astype(int)
            tp = ((yp == 1) & (y_true_binary == 1)).sum()
            fp = ((yp == 1) & (y_true_binary == 0)).sum()
            fn = ((yp == 0) & (y_true_binary == 1)).sum()
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0
            marker = " <-- 当前" if abs(thr_val - self.threshold) < 1e-6 else ""
            print(f"  {thr_val:10.4f}  {p:9.3f}  {r:6.3f}  {f:6.3f}{marker}")

        # --- Stage-2 评估: 仅真实周跳样本 ---
        true_slip_mask = y_test > 0
        n_true_slips = true_slip_mask.sum()
        s2_report = ""
        if n_true_slips > 0:
            y_true_slip = y_test[true_slip_mask]
            y_pred_slip = y_pred[true_slip_mask]
            print(f"\n--- Stage-2 评估: 4分类 (真实周跳: {n_true_slips} 样本) ---")
            present = sorted(set(y_true_slip) | set(y_pred_slip))
            names = [self.CLASS_NAMES[i] if i < len(self.CLASS_NAMES)
                     else f'Class{i}' for i in present]
            s2_report = classification_report(
                y_true_slip, y_pred_slip,
                labels=present, target_names=names)
            print(s2_report)
        else:
            print("  Stage-2: 测试集无真实周跳样本，跳过")

        # --- 整体 5分类评估 ---
        print(f"--- 整体 5分类评估 ---")
        present_all = sorted(set(y_test) | set(y_pred))
        names_all = [self.CLASS_NAMES[i] if i < len(self.CLASS_NAMES)
                     else f'Class{i}' for i in present_all]
        overall_report = classification_report(
            y_test, y_pred,
            labels=present_all, target_names=names_all)
        print(overall_report)

        cm = confusion_matrix(y_test, y_pred, labels=present_all)
        print("混淆矩阵:")
        print(cm)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"\n整体准确率: {acc:.4f}, 加权F1: {f1:.4f}")

        return s1_report, s2_report, overall_report, cm

    def feature_importance(self, top_n=15):
        """返回两个阶段的特征重要性"""
        results = {}
        for stage_name, model in [('Stage-1(Normal/Slip)', self.stage1),
                                   ('Stage-2(SlipType)', self.stage2)]:
            importances = model.feature_importances_
            if self.feature_names is not None:
                indices = np.argsort(importances)[::-1][:top_n]
                print(f"\n--- {stage_name} Top-{top_n} 特征重要性 ---")
                for rank, idx in enumerate(indices):
                    print(f"  {rank+1:2d}. {self.feature_names[idx]:20s}: "
                          f"{importances[idx]:.4f}")
                results[stage_name] = [
                    (self.feature_names[i], importances[i]) for i in indices]
            else:
                results[stage_name] = importances
        return results


class OutlierDetector:
    """粗差检测器 (Random Forest + Isolation Forest)"""

    def __init__(self, random_state=42):
        self.rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
        self.iso = IsolationForest(
            n_estimators=200,
            contamination=0.02,  # 预期2%异常
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_names = None

    def train(self, X_train, y_train, feature_names=None):
        self.feature_names = feature_names
        print(f"\n{'='*60}")
        print("训练粗差检测器")

        # RF (有监督)
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"  训练集: {len(X_train)} 样本")
        for u, c in zip(unique, counts):
            name = 'Normal' if u == 0 else 'Outlier'
            print(f"    {name}: {c} ({c/len(y_train)*100:.1f}%)")

        self.rf.fit(X_train, y_train)
        print("  Random Forest 训练完成")

        # Isolation Forest (无监督)
        self.iso.fit(X_train)
        print("  Isolation Forest 训练完成")

    def evaluate(self, X_test, y_test):
        # RF 评估
        y_pred_rf = self.rf.predict(X_test)
        print(f"\n--- RF粗差检测评估 (测试集: {len(X_test)} 样本) ---")
        report = classification_report(y_test, y_pred_rf,
                                       target_names=['Normal', 'Outlier'])
        print(report)

        # Isolation Forest 评估
        y_pred_iso_raw = self.iso.predict(X_test)
        # IF: 1=normal, -1=anomaly -> 转换为 0/1
        y_pred_iso = np.where(y_pred_iso_raw == -1, 1, 0)
        print("--- Isolation Forest 对照评估 ---")
        report_iso = classification_report(y_test, y_pred_iso,
                                           target_names=['Normal', 'Outlier'])
        print(report_iso)

        return report

    def feature_importance(self, top_n=15):
        importances = self.rf.feature_importances_
        if self.feature_names is not None:
            indices = np.argsort(importances)[::-1][:top_n]
            print(f"\n--- 粗差检测 Top-{top_n} 特征重要性 ---")
            for rank, idx in enumerate(indices):
                print(f"  {rank+1:2d}. {self.feature_names[idx]:20s}: {importances[idx]:.4f}")
            return [(self.feature_names[i], importances[i]) for i in indices]
        return importances


# =============================================================================
# 可视化
# =============================================================================
def plot_feature_importance(importances, title, save_path):
    """绘制特征重要性条形图"""
    names = [x[0] for x in importances]
    values = [x[1] for x in importances]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(names)), values[::-1], color='steelblue')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel('Feature Importance')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  图片已保存: {save_path}")


def plot_timeline(df, y_true_cs, y_pred_cs, y_true_out, y_pred_out, save_path):
    """时间序列可视化: 真实标签 vs 模型预测"""
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    epochs = df['epoch_idx'].values

    # 周跳标签 (真实)
    ax = axes[0]
    ax.scatter(epochs, y_true_cs, c=y_true_cs, cmap='Set1', s=2, alpha=0.6)
    ax.set_ylabel('True CS Label')
    ax.set_title('Cycle Slip: True Labels (0=Normal, 1=Ion, 2=MW, 3=Div, 4=New)')

    # 周跳标签 (预测)
    ax = axes[1]
    ax.scatter(epochs, y_pred_cs, c=y_pred_cs, cmap='Set1', s=2, alpha=0.6)
    ax.set_ylabel('Pred CS Label')
    ax.set_title('Cycle Slip: RF Predictions')

    # 粗差标签 (真实)
    ax = axes[2]
    ax.scatter(epochs, y_true_out, c=y_true_out, cmap='Set1', s=2, alpha=0.6)
    ax.set_ylabel('True Outlier')
    ax.set_title('Code Outlier: True Labels (0=Normal, 1=Outlier)')

    # 粗差标签 (预测)
    ax = axes[3]
    ax.scatter(epochs, y_pred_out, c=y_pred_out, cmap='Set1', s=2, alpha=0.6)
    ax.set_ylabel('Pred Outlier')
    ax.set_xlabel('Epoch Index')
    ax.set_title('Code Outlier: RF Predictions')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  图片已保存: {save_path}")


def plot_label_comparison(df, save_path):
    """日志标签 vs 阈值标签对比可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    epochs = df['epoch_idx'].values

    # 周跳: 日志标签
    ax = axes[0, 0]
    ax.scatter(epochs, df['cs_label_log'].values,
               c=df['cs_label_log'].values, cmap='Set1', s=2, alpha=0.6)
    ax.set_ylabel('CS Label')
    ax.set_title('Cycle Slip: C++ Log Labels')

    # 周跳: 阈值标签
    ax = axes[0, 1]
    ax.scatter(epochs, df['cs_label_thresh'].values,
               c=df['cs_label_thresh'].values, cmap='Set1', s=2, alpha=0.6)
    ax.set_ylabel('CS Label')
    ax.set_title('Cycle Slip: Threshold Labels')

    # 粗差: 日志标签
    ax = axes[1, 0]
    ax.scatter(epochs, df['out_label_log'].values,
               c=df['out_label_log'].values, cmap='Set1', s=2, alpha=0.6)
    ax.set_ylabel('Out Label')
    ax.set_xlabel('Epoch Index')
    ax.set_title('Outlier: C++ Log Labels')

    # 粗差: 阈值标签
    ax = axes[1, 1]
    ax.scatter(epochs, df['out_label_thresh'].values,
               c=df['out_label_thresh'].values, cmap='Set1', s=2, alpha=0.6)
    ax.set_ylabel('Out Label')
    ax.set_xlabel('Epoch Index')
    ax.set_title('Outlier: Threshold Labels')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  标签对比图已保存: {save_path}")


# =============================================================================
# 主函数
# =============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ===== 1. 加载数据 =====
    print("=" * 60)
    print("STEP 1: 加载数据")
    print("=" * 60)

    ext_A = GNSSFeatureExtractor(RINEX_A, SP3_A, label='A')
    ext_B = GNSSFeatureExtractor(RINEX_B, SP3_B, label='B')
    ext_A.load_data()
    ext_B.load_data()

    # ===== 2. 特征提取 =====
    print("\n" + "=" * 60)
    print("STEP 2: 特征提取")
    print("=" * 60)

    df_A = ext_A.extract_features()
    df_B = ext_B.extract_features()

    # 保存单星特征
    df_A.to_csv(os.path.join(OUTPUT_DIR, 'features_A.csv'), index=False)
    df_B.to_csv(os.path.join(OUTPUT_DIR, 'features_B.csv'), index=False)
    print(f"  特征已保存到 {OUTPUT_DIR}")

    # ===== 3. 星间单差 =====
    print("\n" + "=" * 60)
    print("STEP 3: 计算星间单差特征")
    print("=" * 60)

    df = compute_single_difference_features(df_A, df_B)

    # ===== 4. 解析C++日志标签 =====
    print("\n" + "=" * 60)
    print("STEP 4: 解析C++日志标签")
    print("=" * 60)

    log_parser = DiffLogParser(LOG_FILE)
    log_labels = log_parser.parse()

    # 合并日志标签到特征DataFrame
    df = merge_log_labels(df, log_parser, tolerance_sec=3.0)

    # 同时生成阈值标签作为对照
    cs_labels_thresh = generate_cycle_slip_labels(df)
    out_labels_thresh = generate_outlier_labels(df)
    df['cs_label_thresh'] = cs_labels_thresh
    df['out_label_thresh'] = out_labels_thresh

    # 使用日志标签作为训练标签
    df['cs_label'] = df['cs_label_log']
    df['out_label'] = df['out_label_log']

    # 标签统计
    print("\n日志周跳标签分布:")
    for i, name in enumerate(CycleSlipClassifier.CLASS_NAMES):
        cnt = (df['cs_label_log'] == i).sum()
        print(f"  {name}: {cnt} ({cnt/len(df)*100:.2f}%)")

    print("\n日志粗差标签分布:")
    for i, name in enumerate(['Normal', 'Outlier']):
        cnt = (df['out_label_log'] == i).sum()
        print(f"  {name}: {cnt} ({cnt/len(df)*100:.2f}%)")

    # 对比: 日志标签 vs 阈值标签
    print("\n--- 日志标签 vs 阈值标签对比 ---")
    cs_agree = (df['cs_label_log'] == df['cs_label_thresh']).sum()
    out_agree = (df['out_label_log'] == df['out_label_thresh']).sum()
    print(f"  周跳标签一致率: {cs_agree}/{len(df)} ({cs_agree/len(df)*100:.1f}%)")
    print(f"  粗差标签一致率: {out_agree}/{len(df)} ({out_agree/len(df)*100:.1f}%)")

    # 交叉表
    print("\n  周跳交叉表 (行=日志, 列=阈值):")
    cs_cross = pd.crosstab(df['cs_label_log'], df['cs_label_thresh'],
                           rownames=['Log'], colnames=['Thresh'])
    print(cs_cross.to_string())

    print("\n  粗差交叉表 (行=日志, 列=阈值):")
    out_cross = pd.crosstab(df['out_label_log'], df['out_label_thresh'],
                            rownames=['Log'], colnames=['Thresh'])
    print(out_cross.to_string())

    # 保存完整数据集
    df.to_csv(os.path.join(OUTPUT_DIR, 'dataset_full.csv'), index=False)

    # ===== 5. 准备训练/测试特征 =====
    print("\n" + "=" * 60)
    print("STEP 5: 准备训练集")
    print("=" * 60)

    # 选择特征列 (数值型，排除标识列和标签列)
    feature_cols = [
        'S1_A', 'S2_A', 'S1_B', 'S2_B',
        'LI_A', 'LI_B', 'MW_A', 'MW_B',
        'dLI_A', 'dLI_B', 'dMW_A', 'dMW_B',
        'dPC_A', 'dPC_B', 'dLC_A', 'dLC_B',
        'div_A', 'div_B',
        'MW_mean_A', 'MW_std_A', 'MW_norm_A',
        'MW_mean_B', 'MW_std_B', 'MW_norm_B',
        'P1_minus_L1_A', 'P1_minus_L1_B',
        'GF_code_A', 'GF_code_B',
        'code_SD_res', 'code_SD_norm',
        'is_new_sat_A',
    ]

    # 确认所有特征列都存在
    available_cols = [c for c in feature_cols if c in df.columns]
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        print(f"  警告: 以下特征列不存在: {missing_cols}")
    feature_cols = available_cols
    print(f"  使用 {len(feature_cols)} 个特征: {feature_cols}")

    # 差分特征在弧段首历元为NaN，用0填充 (表示"无变化信息")
    diff_cols = [c for c in feature_cols if c.startswith('d') or c == 'div_A' or c == 'div_B']
    for c in diff_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)
    # 其余NaN用该列中位数填充
    for c in feature_cols:
        if c in df.columns and df[c].isna().any():
            median_val = df[c].median()
            if np.isnan(median_val):
                median_val = 0.0
            df[c] = df[c].fillna(median_val)

    df_clean = df.dropna(subset=feature_cols).copy()
    print(f"  有效记录: {len(df_clean)} (原: {len(df)})")

    # 按时间80/20分割
    n_total = len(df_clean)
    n_train = int(n_total * 0.8)

    X_all = df_clean[feature_cols].values
    y_cs_all = df_clean['cs_label'].values
    y_out_all = df_clean['out_label'].values

    X_train, X_test = X_all[:n_train], X_all[n_train:]
    y_cs_train, y_cs_test = y_cs_all[:n_train], y_cs_all[n_train:]
    y_out_train, y_out_test = y_out_all[:n_train], y_out_all[n_train:]

    print(f"  训练集: {len(X_train)}, 测试集: {len(X_test)}")

    # ===== 6. 训练模型 =====
    print("\n" + "=" * 60)
    print("STEP 6: 训练模型")
    print("=" * 60)

    # 周跳分类器 (二阶段)
    cs_clf = TwoStageCycleSlipClassifier()
    cs_clf.train(X_train, y_cs_train, feature_names=feature_cols)
    s1_report, s2_report, cs_report, cs_cm = cs_clf.evaluate(X_test, y_cs_test)
    cs_imp = cs_clf.feature_importance()

    # 粗差检测器
    out_det = OutlierDetector()
    out_det.train(X_train, y_out_train, feature_names=feature_cols)
    out_report = out_det.evaluate(X_test, y_out_test)
    out_imp = out_det.feature_importance()

    # ===== 7. 保存结果 =====
    print("\n" + "=" * 60)
    print("STEP 7: 保存结果")
    print("=" * 60)

    # 保存报告
    with open(os.path.join(OUTPUT_DIR, 'cycle_slip_report.txt'), 'w',
              encoding='utf-8') as f:
        f.write("Two-Stage Cycle Slip Classification Report (C++ Log Labels)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Log file: {os.path.basename(LOG_FILE)}\n")
        f.write(f"Log events: {sum(log_parser.stats.values())}\n")
        for k, v in log_parser.stats.items():
            if v > 0:
                f.write(f"  {k}: {v}\n")
        f.write("\n--- Stage-1: Normal vs AnySlip ---\n")
        f.write(s1_report)
        f.write("\n--- Stage-2: Slip Type ---\n")
        f.write(s2_report if s2_report else "No slip samples in test set\n")
        f.write("\n--- Overall 5-class ---\n")
        f.write(cs_report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cs_cm))
    print("  周跳分类报告已保存")

    with open(os.path.join(OUTPUT_DIR, 'outlier_report.txt'), 'w',
              encoding='utf-8') as f:
        f.write("Outlier Detection Report (C++ Log Labels)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Log file: {os.path.basename(LOG_FILE)}\n")
        f.write(out_report)
    print("  粗差检测报告已保存")

    # 保存可视化
    if cs_imp:
        for stage_name, imp_list in cs_imp.items():
            safe_name = stage_name.replace('/', '-').replace('(', '').replace(')', '')
            plot_feature_importance(
                imp_list,
                f'Cycle Slip - {stage_name}',
                os.path.join(OUTPUT_DIR, f'feature_importance_cs_{safe_name}.png'))
    if out_imp:
        plot_feature_importance(out_imp, 'Outlier Detection - Feature Importance',
                                os.path.join(OUTPUT_DIR, 'feature_importance_outlier.png'))

    # 时间序列可视化
    y_pred_cs_all = cs_clf.predict(X_all)
    y_pred_out_all = out_det.rf.predict(X_all)
    plot_timeline(df_clean, y_cs_all, y_pred_cs_all,
                  y_out_all, y_pred_out_all,
                  os.path.join(OUTPUT_DIR, 'timeline_predictions.png'))

    # 标签对比可视化
    plot_label_comparison(df_clean, os.path.join(OUTPUT_DIR, 'label_comparison.png'))

    # 保存模型
    joblib.dump(cs_clf.stage1, os.path.join(OUTPUT_DIR, 'cs_stage1_rf.joblib'))
    joblib.dump(cs_clf.stage2, os.path.join(OUTPUT_DIR, 'cs_stage2_rf.joblib'))
    joblib.dump(out_det.rf, os.path.join(OUTPUT_DIR, 'outlier_rf.joblib'))
    joblib.dump(out_det.iso, os.path.join(OUTPUT_DIR, 'outlier_iforest.joblib'))
    print("  模型已保存 (joblib)")

    # 数据统计摘要
    print("\n" + "=" * 60)
    print("完成! 输出文件:")
    print("=" * 60)
    for f in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(fpath)
        print(f"  {f:40s} {size/1024:.1f} KB")


if __name__ == '__main__':
    main()
