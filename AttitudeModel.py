import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import TimeSystem
import matplotlib.pyplot as plt
import json
import math

class Attitude:
    def __init__(self, q=None):
        """
        初始化姿态对象。
        参数 q: 四元数列表 [q1, q2, q3, q4]，通常 q4 为标量 (w)。
               (这是 Scipy Rotation 库的默认接受格式 [x, y, z, w])
        """
        self.q = q  
        if self.q is not None:
            self.rotation_matrix = self.quat2matrix(self.q)
            self.euler_angles = self.quat2euler(self.q)
    
    def __repr__(self):
        return f"Attitude(q={self.q})"
    
    def quat2matrix(self, q):
        """Convert quaternion to rotation matrix."""
        # 注意: R.from_quat 默认输入格式是 [x, y, z, w]
        r = R.from_quat([q[0], q[1], q[2], q[3]])
        return r.as_matrix()
    
    def quat2euler(self, q):
        """Convert quaternion to Euler angles (roll, pitch, yaw).
        Returns angles in degrees.
        """
        r = R.from_quat([q[0], q[1], q[2], q[3]])
        return r.as_euler('xyz', degrees=True)
    
    def get_euler_angles(self):
        return self.euler_angles
    
    def get_rotation_matrix(self):
        return self.rotation_matrix
    
def RotM_Ecef_J2k(t : TimeSystem.TimeSystem):
    mjd = t.mjd
    j2000_mjd = 51544.5
    gmst = math.radians((280.46061837 + 360.98564736629 * (mjd - j2000_mjd)) % 360.0)
    c, s = math.cos(gmst), math.sin(gmst)

    # 旋转矩阵
    return np.array([
        [ c,  s, 0.0],
        [-s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])

def inertial2rtn(pos, vel, vector_inertial):
    """
    将惯性系 (Inertial Frame) 下的向量转换到轨道 RTN 系。
    
    参数:
    - pos: numpy array [X, Y, Z] 卫星在惯性系下的位置向量
    - vel: numpy array [Vx, Vy, Vz] 卫星在惯性系下的速度向量
    - vector_inertial: numpy array [Vx, Vy, Vz] 需要转换的惯性系向量
    
    返回:
    - vector_rtn: numpy array [R, T, N] 转换到 RTN 系的向量
    """
    # 确保输入是 numpy 数组
    r = np.array(pos, dtype=float)
    v = np.array(vel, dtype=float)
    vec_i = np.array(vector_inertial, dtype=float)
    
    # 1. 径向 (Radial, R): 位置矢量的单位向量
    u_r = r / np.linalg.norm(r)
    
    # 2. 法向 (Normal, N): 角动量方向 (r x v) 的单位向量
    h = np.cross(r, v)
    u_n = h / np.linalg.norm(h)
    
    # 3. 横向/沿轨 (Transverse, T): n x r，完成右手坐标系
    u_t = np.cross(u_n, u_r)
    
    # 构建旋转矩阵 (从惯性系到 RTN 系)
    # 矩阵的每一行就是对应的基向量
    R_inertial_to_rtn = np.vstack([u_r, u_t, u_n])
    
    # 进行转换
    vector_rtn = R_inertial_to_rtn @ vec_i
    
    return vector_rtn


# ================== 文件读取辅助函数 ==================

def read_attitude_file(filepath):
    """Reads attitude data from a specified file and returns a DataFrame.
    columns: year, month, day, hour, minute, second, q1, q2, q3, q4
    """
    tmpdata = pd.read_csv(filepath, delim_whitespace=True, header=None,
                          names=['year', 'month', 'day', 'hour', 'minute', 'second', 
                                 'q1', 'q2', 'q3', 'q4'])
    tmpdata[['year', 'month', 'day', 'hour', 'minute']] = tmpdata[['year', 'month', 'day', 'hour', 'minute']].astype(int)
    tmpdata['second'] = tmpdata['second'].astype(float)
    
    # 将时间转化为之前重写的高级 TimeSystem 对象
    tmpdata['timesystem'] = tmpdata.apply(TimeSystem.make_ts, axis=1)
    
    # 构建 Attitude 对象
    tmpdata['attitude'] = tmpdata.apply(lambda row: Attitude(q=[row['q1'], row['q2'], row['q3'], row['q4']]), axis=1)
    
    data = tmpdata[['timesystem', 'attitude', 'q1', 'q2', 'q3', 'q4']].reset_index(drop=True)
    return data

def read_empacc_file(filepath):
    """读取经验加速度等数据文件"""
    data = pd.read_csv(filepath, delim_whitespace=True, header=None,
                       names=['year', 'month', 'day', 'hour', 'minute', 'second', 'timeofday',
                              'Cd', 'Cr', 'bias_x', 'bias_y', 'bias_z',
                              'acc_x', 'acc_y', 'acc_z'])
    data[['year', 'month', 'day', 'hour', 'minute']] = data[['year', 'month', 'day', 'hour', 'minute']].astype(int)
    data['second'] = data['second'].astype(float) # 确保 second 是浮点数，防止精度丢失
    data['timesystem'] = data.apply(TimeSystem.make_ts, axis=1)
    return data 



# # 移除了重复的独立 quat2matrix 和 quat2euler 函数
# if __name__ == "__main__":
#     configpath = "attitude_config.json"
#     with open(configpath, 'r') as f:
#         config = json.load(f)
    
#     filepath = config["attitude_file"]
#     attitude_data = read_attitude_file(filepath)
#     empacc_filepath = config["empacc_file"]
#     empacc_data = read_empacc_file(empacc_filepath)

#     attitude_data, empacc_data = TimeSystem.align_dataframes_on_times(attitude_data, empacc_data)

#     acc_body_list = []

#     print(f"Aligned data length: {len(attitude_data)}, {len(empacc_data)}")
#     print(attitude_data.tail())
#     print(empacc_data.tail())

#     for i in range(len(attitude_data)):
#         ts = attitude_data.loc[i, 'timesystem']
#         q = attitude_data.loc[i, ['q1', 'q2', 'q3', 'q4']].values
#         acc = empacc_data.loc[i, ['acc_x', 'acc_y', 'acc_z']].values

#         R_matrix = Attitude.quat2matrix(q)
#         acc_body = R_matrix @ acc  # Transform acceleration to body frame

#         # print(f"Time: {ts.to_str()}, Acceleration in body frame: {acc_body}")
#         acc_body_list.append(acc_body)
#     acc_body_array = np.array(acc_body_list)
#     # 分为三幅图
#     plt.figure(figsize=(12, 8))
#     plt.subplot(3, 1, 1)
#     plt.plot(acc_body_array[:, 0], label='Acc X (body frame)', color='r')
#     plt.xlabel('Time Index')
#     plt.ylabel('Acceleration (m/s²)')
#     plt.title('Acceleration in Body Frame Over Time - X Component')
#     plt.legend()
#     plt.subplot(3, 1, 2)
#     plt.plot(acc_body_array[:, 1], label='Acc Y (body frame)', color='g')
#     plt.xlabel('Time Index')
#     plt.ylabel('Acceleration (m/s²)')
#     plt.title('Acceleration in Body Frame Over Time - Y Component')
#     plt.legend()
#     plt.subplot(3, 1, 3)
#     plt.plot(acc_body_array[:, 2], label='Acc Z (body frame)', color='b')
#     plt.xlabel('Time Index')
#     plt.ylabel('Acceleration (m/s²)')
#     plt.title('Acceleration in Body Frame Over Time - Z Component')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
