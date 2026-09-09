import numpy as np
from dataclasses import dataclass, field

@dataclass
class PCV_T:
    """完美对应 RTKLIB 中的 pcv_t 结构体"""
    sat: str = ""              # 卫星编号 (为空代表是接收机天线)
    type: str = ""             # 天线类型
    code: str = ""             # 序列号或卫星代码
    # 用字典来替代 C 语言中的多维数组 off[NFREQ][3] 和 var[NFREQ][19]
    # key: 频段标号(如 'G01', 'G02'), value: 数据列表/数组
    off: dict = field(default_factory=dict) # PCO [x/e, y/n, z/u] (单位: 米)
    var: dict = field(default_factory=dict) # PCV NOAZI 19个值 (单位: 米)
    zen1: float = 0.0  # 离地角起始值 (通常是 0.0度)
    zen2: float = 90.0 # 离地角结束值 (卫星通常是 20度，接收机通常是 90度)
    dzen: float = 5.0  # 离地角步长 (卫星通常是1度，接收机通常是 5.0度)

    def __repr__(self):
        return (f"PCV_T(sat='{self.sat}', type='{self.type}', code='{self.code}', "
                f"zen1={self.zen1}, zen2={self.zen2}, dzen={self.dzen})")

class ATXParserRTK:
    def __init__(self, filepath):
        self.filepath = filepath
        self.pcvs = [] # 存储所有解析出来的天线数据
        self._parse()

    def _parse(self):
        """完全模仿 readantex 的状态机逻辑"""
        state = 0
        current_pcv = None
        current_freq = None

        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if len(line) < 60 or "COMMENT" in line[60:]:
                    continue
                
                # 状态机：开始读取一个天线块
                if "START OF ANTENNA" in line[60:]:
                    current_pcv = PCV_T()
                    state = 1
                    
                # 状态机：结束读取，存入列表
                if "END OF ANTENNA" in line[60:]:
                    self.pcvs.append(current_pcv)
                    state = 0
                    
                if not state:
                    continue

                if "ZEN1 / ZEN2 / DZEN" in line[60:]:
                    parts = line[:60].split()
                    current_pcv.zen1 = float(parts[0])
                    current_pcv.zen2 = float(parts[1])
                    current_pcv.dzen = float(parts[2])

                # --- 提取头信息 ---
                if "TYPE / SERIAL NO" in line[60:]:
                    current_pcv.type = line[0:20].strip()
                    current_pcv.code = line[20:40].strip()
                    # 判断是卫星还是接收机 (类似 RTKLIB 的逻辑)
                    if len(current_pcv.code) >= 3 and current_pcv.code[0] in ['G', 'R', 'E', 'C', 'J', 'S']:
                        current_pcv.sat = current_pcv.code[:3]
                    else:
                        current_pcv.sat = "" # 接收机天线

                # --- 提取频段 ---
                elif "START OF FREQUENCY" in line[60:]:
                    current_freq = line[0:10].strip()
                
                elif "END OF FREQUENCY" in line[60:]:
                    current_freq = None

                # --- 提取 PCO (转换为米) ---
                elif "NORTH / EAST / UP" in line[60:] and current_freq:
                    parts = line[:60].split()
                    if len(parts) >= 3:
                        neu = [float(parts[0]) / 1000.0, float(parts[1]) / 1000.0, float(parts[2]) / 1000.0]
                        # 完美复刻 RTKLIB 的 x/e, y/n 互换逻辑
                        if current_pcv.sat: # 是卫星
                            current_pcv.off[current_freq] = np.array([neu[0], neu[1], neu[2]]) # X, Y, Z
                        else: # 是接收机
                            current_pcv.off[current_freq] = np.array([neu[1], neu[0], neu[2]]) # E, N, U

                # --- 提取 PCV NOAZI (转换为米) ---
                elif "NOAZI" in line and current_freq:
                    parts = line[8:].split() # RTKLIB从+8开始读
                    var_values = []
                    for val in parts:
                        var_values.append(float(val) / 1000.0)
                        
                    # 如果数据不足19个，像RTKLIB一样用最后一个值补齐 (这里仅作展示，通常ATX文件都是全的)
                    while len(var_values) < 19:
                        var_values.append(var_values[-1] if var_values else 0.0)
                        
                    current_pcv.var[current_freq] = np.array(var_values[:19])

    def get_pcv(self, sat_id):
        """便捷检索方法：根据卫星号返回 PCV_T 对象"""
        for pcv in self.pcvs:
            if pcv.sat == sat_id:
                return pcv
        return None
    
    def get_receiver_pcv(self, ant_type):
        """
        便捷检索方法：根据接收机天线型号返回 PCV_T 对象
        :param ant_type: 从 RINEX 头读取的天线型号，如 "TRM59800.00     SCIS"
        """
        target_type = ant_type.rstrip() # 去除右侧空格对齐格式
        for pcv in self.pcvs:
            # 必须满足 sat 为空（是接收机），且 type 匹配
            if not pcv.sat and pcv.type == target_type:
                print("Find matching receiver antenna in ATX:", pcv.type)
                return pcv
        return None


def correct_receiver_antenna(obs_raw, elevation_deg, azimuth_deg, 
                             delta_hen, pco_enu, pcv_noazi_array, zen1=0.0, dzen=5.0):
    """
    修正接收机天线误差 (将观测值归化到测站标石中心)
    
    :param obs_raw: 原始观测值 (米)
    :param elevation_deg: 卫星高度角 (度)
    :param azimuth_deg: 卫星方位角 (度)
    :param delta_hen: RINEX中的测站偏心距 [H, E, N] (米)
    :param pco_enu: ATX中的天线 PCO [E, N, U] (米)
    :param pcv_noazi_array: ATX中的 NOAZI PCV 数组 (米)
    :param zen1: PCV 数组的起始天顶距 (通常是 0.0度)
    :param dzen: PCV 数组的天顶距步长 (通常是 5.0度)
    :return: 修正后的观测值 (米), 观测值的总修正量 (米)
    """
    # 1. 计算总偏心向量 (ENU 坐标系)
    d_E = delta_hen[1] + pco_enu[0]
    d_N = delta_hen[2] + pco_enu[1]
    d_U = delta_hen[0] + pco_enu[2]
    
    # 2. 角度转弧度
    el_rad = np.radians(elevation_deg)
    az_rad = np.radians(azimuth_deg)
    
    # 3. 计算 LOS 投影 (点积)
    # Proj = d_E * cos(el)*sin(az) + d_N * cos(el)*cos(az) + d_U * sin(el)
    projection = (d_E * np.cos(el_rad) * np.sin(az_rad) + 
                  d_N * np.cos(el_rad) * np.cos(az_rad) + 
                  d_U * np.sin(el_rad))
    
    # 4. 计算 PCV (一维线性插值)
    zenith_deg = 90.0 - elevation_deg
    
    # 构造 PCV 的自变量网格 [0, 5, 10, ... 90]
    num_zenith_points = len(pcv_noazi_array)
    zenith_grid = np.array([zen1 + i * dzen for i in range(num_zenith_points)])
    
    # 确保插值在网格范围内
    zenith_query = np.clip(zenith_deg, zenith_grid[0], zenith_grid[-1])
    pcv_val = np.interp(zenith_query, zenith_grid, pcv_noazi_array)
    
    # 5. 应用最终修正
    obs_corrected = obs_raw + projection - pcv_val
    
    return obs_corrected

import numpy as np
import math

def correct_satellite_observation(obs_raw, r_sat, r_rcv, r_sun, pco_body, pcv_noazi_array, zen1=0.0, dzen=1.0):
    """
    修正观测值中的卫星天线误差 (将观测值从 PCo 归化到 CoM)
    
    :param obs_raw: 原始观测值 (伪距或载波相位，单位：米)
    :param r_sat: 卫星 ECEF 坐标 [X, Y, Z] (米)
    :param r_rcv: 接收机 ECEF 坐标 [X, Y, Z] (米)
    :param r_sun: 太阳 ECEF 坐标 [X, Y, Z] (米)
    :param pco_body: ATX 中提取的卫星 PCO [X, Y, Z] (米)
    :param pcv_noazi_array: ATX 中提取的卫星 PCV 数组 (米)
    :param zen1: PCV 数组的起始离地角 (通常为 0.0 度)
    :param dzen: PCV 数组的步长 (通常为 1.0 度或 2.0 度，详见 atx 文件的 DZEN)
    :return: 修正后的观测值 (米), 观测值的总修正量 (米)
    """
    # ---------------------------------------------------------
    # 1. 建立卫星本体坐标系 (名义偏航姿态 Nominal Yaw Steering)
    # ---------------------------------------------------------
    # Z 轴：卫星质心指向地心 (与 r_sat 向量方向相反)
    e_z = -r_sat / np.linalg.norm(r_sat)
    
    # Y 轴：太阳能帆板旋转轴，垂直于卫星、地心、太阳组成的平面
    e_y_raw = np.cross(e_z, r_sun)
    e_y = e_y_raw / np.linalg.norm(e_y_raw)
    
    # X 轴：构成右手坐标系
    e_x = np.cross(e_y, e_z)
    
    # ---------------------------------------------------------
    # 2. 空间几何投影 (计算视线方向)
    # ---------------------------------------------------------
    # 计算视线向量 (Line of Sight): 从【卫星】指向【接收机】
    los_vector = r_rcv - r_sat
    e_los = los_vector / np.linalg.norm(los_vector)
    
    # 将 PCO 从卫星本体坐标系转换到 ECEF 坐标系
    pco_ecef = pco_body[0] * e_x + pco_body[1] * e_y + pco_body[2] * e_z
    
    # 投影到视线方向：点乘计算出 PCo 沿着视线向接收机前进了多少米
    pco_proj = np.dot(pco_ecef, e_los) 
    
    # ---------------------------------------------------------
    # 3. 计算 PCV (离地角一维插值)
    # ---------------------------------------------------------
    pcv_val = 0.0
    if pcv_noazi_array is not None and len(pcv_noazi_array) > 0:
        # 计算离地角 Nadir (Z 轴与视线向量的夹角)
        # 因为 e_los 和 e_z 都是单位向量，点乘即为 cos(nadir)
        cos_nadir = np.clip(np.dot(e_los, e_z), -1.0, 1.0)
        nadir_deg = math.degrees(math.acos(cos_nadir))
        
        # 构造网格并进行一维线性插值
        num_zen = len(pcv_noazi_array)
        zen_grid = np.array([zen1 + i * dzen for i in range(num_zen)])
        nadir_query = np.clip(nadir_deg, zen_grid[0], zen_grid[-1])
        pcv_val = np.interp(nadir_query, zen_grid, pcv_noazi_array)
        
    # ---------------------------------------------------------
    # 4. 应用最终修正
    # ---------------------------------------------------------
    # 核心逻辑：因为测出的距离“短了”，为了恢复为到质心的几何距离，必须加回去。
    obs_corrected = obs_raw + pco_proj + pcv_val
    
    return obs_corrected

def read_atx_correct_sat_rec_ant(obs_raw, atx_file, prn, freq, rec_ecef, sat_ecef, sun_ecef, elevation_deg, azimuth_deg, delta_hen, rec_ant):
    # 适用于单次，需要循环的时候把parser拿出去提高效率

    parser = ATXParserRTK(atx_file)
    prn_freq = f"{prn[0]}{freq}"
    
    # 获取卫星天线数据
    sat_pcv = parser.get_pcv(prn)
    if not sat_pcv:
        raise ValueError(f"未在 ATX 文件中找到卫星 {prn} 的天线数据！")
    sat_pco = sat_pcv.off.get(prn_freq)
    sat_pcv_noazi = sat_pcv.var.get(prn_freq)
    if sat_pco is None or sat_pcv_noazi is None:
        raise ValueError(f"卫星 {prn} {prn_freq} 的 PCO 或 PCV 数据不完整！")
    
    # 获取接收机天线数据
    rx_antenna = parser.get_receiver_pcv(rec_ant)
    if not rx_antenna:
        raise ValueError(f"未在 ATX 文件中找到接收机天线 {rec_ant} 的数据！")
    rec_pco_enu = rx_antenna.off.get(prn_freq)
    rec_pcv_noazi = rx_antenna.var.get(prn_freq)
    if rec_pco_enu is None or rec_pcv_noazi is None:
        raise ValueError(f"接收机天线 {rec_ant} {prn_freq} 的 PCO 或 PCV 数据不完整！")
    
    # 先修正卫星端误差
    print(f"prn={prn}, sat_pco={sat_pco}, sat_pcv_noazi={sat_pcv_noazi}")
    obs_after_sat = correct_satellite_observation(obs_raw, sat_ecef, rec_ecef, sun_ecef,
                                                  sat_pco, sat_pcv_noazi,
                                                  zen1=sat_pcv.zen1, dzen=sat_pcv.dzen)
    
    # 再修正接收机端误差
    obs_after_rec = correct_receiver_antenna(obs_after_sat, elevation_deg, azimuth_deg, delta_hen, 
                                           rec_pco_enu, rec_pcv_noazi,
                                           zen1=rx_antenna.zen1, dzen=rx_antenna.dzen)
    return obs_after_rec

def correct_sat_rec_ant(obs_raw, sat_ecef, rec_ecef, sun_ecef, freq, 
                        sat_pcvt, sat_pcv, rec_pcvt, rec_pcv, elevation_deg, azimuth_deg, delta_hen):
    # 先修正卫星端误差
    if sat_pcvt is not None and sat_pcv is not None:
        sat_pco = sat_pcv.get(freq).get('pco')
        sat_pcv_noazi = sat_pcv.get(freq).get('noazi')
        obs_after_sat = correct_satellite_observation(obs_raw, sat_ecef, rec_ecef, sun_ecef,
                                                    sat_pco, sat_pcv_noazi,
                                                    zen1=sat_pcvt.zen1, dzen=sat_pcvt.dzen)
    else:
        obs_after_sat = obs_raw

    if rec_pcv is not None and rec_pcvt is not None:
        rec_pco_enu = rec_pcv.get(freq).get('pco')
        rec_pcv_noazi = rec_pcv.get(freq).get('noazi')
        obs_after_rec = correct_receiver_antenna(obs_after_sat, elevation_deg, azimuth_deg, delta_hen, 
                                           rec_pco_enu, rec_pcv_noazi,
                                           zen1=rec_pcvt.zen1, dzen=rec_pcvt.dzen)
    else:
        obs_after_rec = obs_after_sat
    return obs_after_rec

# ==========================================
# 简单的调用示例
# ==========================================
# if __name__ == "__main__":
#     # 模拟数据
#     obs = 22000000.0  # 原始伪距 (米)
#     r_sat = np.array([15600000.0, 15600000.0, 15600000.0])
#     r_rcv = np.array([6378137.0, 0.0, 0.0])
#     r_sun = np.array([1.496e11, 0.0, 0.0])
    
#     # 假设你从 ATX 中提取出了 G01 L1 的数据
#     pco_sat = np.array([0.394, 0.0, 1.500])  # 本体坐标系下的 PCO
#     pcv_sat = np.zeros(15) # 简化的 15个 0 的数组，代表 PCV (0~14度)
    
#     obs_new = correct_satellite_observation(obs, r_sat, r_rcv, r_sun, pco_sat, pcv_sat)
    
#     print(f"原始观测值: {obs:.4f} m")
#     print(f"PCO+PCV 修正后: {obs_new:.4f} m")
#     print(f"卫星端总修正补偿量: {obs_new - obs:.4f} m")

# # 测试代码
if __name__ == "__main__":
    parser = ATXParserRTK("../time_sync/igs20.atx")
    g01_antenna = parser.get_pcv("G01")
    print(g01_antenna)


    rinex_antenna = "TRM59800.00     SCIT"
    rx_antenna = parser.get_receiver_pcv(rinex_antenna)
    if rx_antenna:
        print(rx_antenna)
        
        # 提取 GPS L1 频段 (G01) 的 PCO
        pco_l1 = rx_antenna.off.get("G01")
        print(f"L1 频段 PCO (East, North, Up) [米]: {pco_l1}")
        
        # 提取 GPS L1 频段 (G01) 的 PCV NOAZI 数组
        pcv_l1 = rx_antenna.var.get("G01")
        print(f"L1 频段 PCV 数组 (0~90度) [米]:\n{pcv_l1}")
    else:
        print("未在 atx 文件中找到该天线，请检查拼写或空格！")

























# import numpy as np
# from dataclasses import dataclass, field
# from scipy.interpolate import RegularGridInterpolator

# @dataclass
# class PCV_T:
#     sat: str = ""              
#     type: str = ""             
#     # 基础 PCO
#     off: dict = field(default_factory=dict) 
#     # 一维 PCV (与方位角无关的平均值)
#     var_noazi: dict = field(default_factory=dict) 
#     # 二维 PCV (方位角相关网格) -> key: freq, value: 2D numpy array [73, num_zenith]
#     var_azi: dict = field(default_factory=dict)   
    
#     # 网格参数
#     zen1: float = 0.0
#     zen2: float = 14.0 # 卫星通常是 14度，接收机通常是 90度
#     dzen: float = 1.0  # 离地角步长
#     dazi: float = 5.0  # 方位角步长

# class ATXParserPro:
#     def __init__(self, filepath):
#         self.filepath = filepath
#         self.pcvs = []
#         self._parse()

#     def _parse(self):
#         state = 0
#         current_pcv = None
#         current_freq = None
#         current_azi_idx = 0 # 记录当前读到了哪个方位角

#         with open(self.filepath, 'r', encoding='utf-8') as f:
#             for line in f:
#                 if len(line) < 60 or "COMMENT" in line[60:]:
#                     continue
                
#                 if "START OF ANTENNA" in line[60:]:
#                     current_pcv = PCV_T()
#                     state = 1
#                 if "END OF ANTENNA" in line[60:]:
#                     self.pcvs.append(current_pcv)
#                     state = 0
                    
#                 if not state:
#                     continue

#                 if "TYPE / SERIAL NO" in line[60:]:
#                     current_pcv.type = line[0:20].strip()
#                     if len(current_pcv.type) > 0 and current_pcv.type[0] in ['G', 'R', 'E', 'C', 'J', 'S']:
#                          current_pcv.sat = current_pcv.type
                         
#                 # 提取网格参数
#                 elif "ZEN1 / ZEN2 / DZEN" in line[60:]:
#                     parts = line[:60].split()
#                     current_pcv.zen1, current_pcv.zen2, current_pcv.dzen = map(float, parts[:3])
#                 elif "DAZI" in line[60:]:
#                     current_pcv.dazi = float(line[:60].split()[0])

#                 elif "START OF FREQUENCY" in line[60:]:
#                     current_freq = line[0:10].strip()
#                     current_azi_idx = 0
#                 elif "END OF FREQUENCY" in line[60:]:
#                     current_freq = None

#                 elif "NORTH / EAST / UP" in line[60:] and current_freq:
#                     neu = [float(x) / 1000.0 for x in line[:60].split()[:3]]
#                     if current_pcv.sat: 
#                         current_pcv.off[current_freq] = np.array([neu[0], neu[1], neu[2]])
#                     else:
#                         current_pcv.off[current_freq] = np.array([neu[1], neu[0], neu[2]])

#                 elif "NOAZI" in line[60:] and current_freq:
#                     var_values = [float(x) / 1000.0 for x in line[8:60].split()]
#                     current_pcv.var_noazi[current_freq] = np.array(var_values)
                    
#                     # 提前初始化二维网格，如果是方位角相关的天线 (DAZI != 0)
#                     if current_pcv.dazi > 0:
#                         num_zen = int((current_pcv.zen2 - current_pcv.zen1) / current_pcv.dzen) + 1
#                         num_azi = int(360.0 / current_pcv.dazi) + 1 # 通常是 73 行 (0度到360度)
#                         current_pcv.var_azi[current_freq] = np.zeros((num_azi, num_zen))

#                 # 提取方位角相关的 PCV 数据 (行首通常是 0.0, 5.0, 10.0...)
#                 elif current_freq and current_pcv.dazi > 0:
#                     try:
#                         # 检查行首是否是一个方位角数值
#                         azi_val = float(line[:8].strip())
#                         # 确保这确实是数据行
#                         var_values = [float(x) / 1000.0 for x in line[8:60].split()]
                        
#                         # 存入二维网格对应的行
#                         if current_azi_idx < current_pcv.var_azi[current_freq].shape[0]:
#                             # 长度截取保护，防止越界
#                             limit = min(len(var_values), current_pcv.var_azi[current_freq].shape[1])
#                             current_pcv.var_azi[current_freq][current_azi_idx, :limit] = var_values[:limit]
#                             current_azi_idx += 1
#                     except ValueError:
#                         pass # 如果不是数值开头的行，直接跳过

# import math

# def compute_full_antenna_correction(pcv_obj, freq_id, r_sat, r_rcv, r_sun):
#     """
#     计算包含 PCO 投影和方位角相关 PCV 的总修正量
#     """
#     # 1. 构建卫星本体坐标系 (Nominal Yaw Steering)
#     e_z = -r_sat / np.linalg.norm(r_sat)
#     e_y_raw = np.cross(e_z, r_sun)
#     e_y = e_y_raw / np.linalg.norm(e_y_raw)
#     e_x = np.cross(e_y, e_z)
    
#     # 2. 视线单位向量 (ECEF)
#     los_vector = r_rcv - r_sat
#     e_los = los_vector / np.linalg.norm(los_vector)
    
#     # ---------------------------------------------------
#     # PCO 修正部分 (与之前一致)
#     # ---------------------------------------------------
#     pco_body = pcv_obj.off.get(freq_id, np.array([0.0, 0.0, 0.0]))
#     pco_ecef = pco_body[0] * e_x + pco_body[1] * e_y + pco_body[2] * e_z
#     pco_corr = np.dot(pco_ecef, e_los)
    
#     # ---------------------------------------------------
#     # PCV 修正部分 (新增方位角逻辑)
#     # ---------------------------------------------------
#     pcv_corr = 0.0
    
#     # 将视线向量 e_los 转换到卫星本体坐标系下 [x, y, z]
#     # 这是最关键的一步，我们需要知道接收机在卫星眼中的相对位置
#     los_body_x = np.dot(e_los, e_x)
#     los_body_y = np.dot(e_los, e_y)
#     los_body_z = np.dot(e_los, e_z)
    
#     # 计算离地角 Nadir (Z轴与视线向量的夹角，范围 0 到 180度)
#     # 因为 e_los 和 e_z 都是单位向量，点乘即为 cos(nadir)
#     cos_nadir = np.clip(np.dot(e_los, e_z), -1.0, 1.0)
#     nadir_deg = math.degrees(math.acos(cos_nadir))
    
#     # 检查当前频率是否有非对称方位角 PCV 数据
#     has_azi_pcv = (freq_id in pcv_obj.var_azi) and (pcv_obj.dazi > 0)
    
#     if has_azi_pcv:
#         # 计算卫星本体坐标系下的方位角 Azimuth (X-Y 平面的夹角)
#         # atan2 返回 -pi 到 pi，转换到 0 到 360度
#         azi_rad = math.atan2(los_body_y, los_body_x)
#         azi_deg = math.degrees(azi_rad)
#         if azi_deg < 0:
#             azi_deg += 360.0
            
#         # 准备双线性插值的网格坐标
#         zen_grid = np.arange(pcv_obj.zen1, pcv_obj.zen2 + pcv_obj.dzen/2, pcv_obj.dzen)
#         azi_grid = np.arange(0.0, 360.0 + pcv_obj.dazi/2, pcv_obj.dazi)
#         pcv_matrix = pcv_obj.var_azi[freq_id]
        
#         # 边界保护：如果离地角超出了ATX定义的范围（比如超过了14度），取边界值
#         nadir_query = np.clip(nadir_deg, pcv_obj.zen1, pcv_obj.zen2)
        
#         # 构造插值器 (Scipy 提供的超高效双线性/多维插值)
#         # 注意矩阵形状和网格的对应关系：这里 matrix 的 shape 是 (num_azi, num_zen)
#         interp_func = RegularGridInterpolator((azi_grid, zen_grid), pcv_matrix)
        
#         # 提取插值结果
#         pcv_corr = interp_func(np.array([[azi_deg, nadir_query]]))[0]
        
#     else:
#         # 如果没有方位角数据，退化为使用 NOAZI 进行一维插值
#         noazi_array = pcv_obj.var_noazi.get(freq_id)
#         if noazi_array is not None:
#             zen_grid = np.arange(pcv_obj.zen1, pcv_obj.zen2 + pcv_obj.dzen/2, pcv_obj.dzen)
#             nadir_query = np.clip(nadir_deg, pcv_obj.zen1, pcv_obj.zen2)
#             pcv_corr = np.interp(nadir_query, zen_grid, noazi_array)

#     # 最终总修正量 = PCO 投影 + PCV 变化
#     return pco_corr + pcv_corr