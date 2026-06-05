import numpy as np 
import pandas as pd 
import datetime as dt
import math 
import AttitudeModel
import TimeSystem


def quat2mat(quat):
    # 标量在后：x, y, z, w
    x, y, z, w = quat
    R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,   2*x*z + 2*y*w],
        [2*x*y + 2*z*w,   1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,   2*y*z + 2*x*w,   1 - 2*x**2 - 2*y**2]
    ])
    return R.T

def ecef2j2k_rotmat(t) -> list[list[float]]:
    """
    计算 ECEF 到 J2000 的 3x3 旋转矩阵（仅绕Z轴旋转）
    :param t: UTC 时间，datetime 对象
    :return: 3x3 旋转矩阵 R，J2000 = R @ ECEF
    """
    # J2000 儒略日
    j2000_jd = 2451545.0

    # datetime 转 儒略日
    y = t.year
    m = t.month
    d = t.day
    h = t.hour + t.minute/60 + (t.second + t.microsecond/1e6)/3600

    if m <= 2:
        y -= 1
        m += 12

    A = y // 100
    B = 2 - A + A//4
    jd = int(365.25*(y+4716)) + int(30.6001*(m+1)) + d + B - 1524.5
    jd += h / 24.0

    # 格林尼治平恒星时（弧度）
    gmst = math.radians((280.46061837 + 360.98564736629 * (jd - j2000_jd)) % 360.0)
    c, s = math.cos(gmst), math.sin(gmst)

    # 旋转矩阵
    return np.array([
        [ c,  s, 0.0],
        [-s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])

def ecef2j2k_rotM_mjd(mjd) -> list[list[float]]:
    """
    计算 ECEF 到 J2000 的 3x3 旋转矩阵（仅绕Z轴旋转）
    :param mjd: Modified Julian Date
    :return: 3x3 旋转矩阵 R，J2000 = R @ ECEF
    """
    # J2000 MJD
    j2000_mjd = 51544.5

    # 格林尼治平恒星时（弧度）
    gmst = math.radians((280.46061837 + 360.98564736629 * (mjd - j2000_mjd)) % 360.0)
    c, s = math.cos(gmst), math.sin(gmst)

    # 旋转矩阵
    return np.array([
        [ c,  s, 0.0],
        [-s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])

quat = np.array([-0.809085, 0.18805878, 0.217267368, -0.51265008])
# M_J2k_B = quat2mat(quat)
# print(M_J2k_B)
attitude = AttitudeModel.Attitude(quat)
M_J2k_B=attitude.get_rotation_matrix().T

t = dt.datetime(2024, 12, 27, 4, 18, 4)
t1 = TimeSystem.TimeSystem()
t1._update_from_datetime(t)
# M_Ecef_J2k = ecef2j2k_rotmat(t)
# print(M_Ecef_J2k)
M_Ecef_J2k=AttitudeModel.RotM_Ecef_J2k(t1)

M_Ecef_B = (M_Ecef_J2k @ M_J2k_B.T).T
M_Ecef_B = (M_J2k_B @ M_Ecef_J2k.T).T
print("M_Ecef_B:", M_Ecef_B)

pos_sat = np.array([21936890, 35990823, 858328.2])
pos_rec = np.array([464301.747, 3728531.384, -5724200.555])
los = pos_sat - pos_rec
los_unit = los / np.linalg.norm(los)
print("LOS unit vector:", los_unit)
los_ecef = M_Ecef_B @ los_unit
print(los_ecef)

pco = np.array([-0.7664, -0.1690, 0.1579])
corr = -np.dot(pco, los_ecef)
print(corr)