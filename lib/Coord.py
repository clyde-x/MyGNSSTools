import numpy as np
import math
from astropy.time import Time
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, get_sun
import astropy.units as u

class GNSSCoordinate:
    """GNSS 空间坐标系转换一体化工具箱"""
    
    # WGS84 椭球体常数
    RE = 6378137.0                   # 地球长半径 (m)
    FE = 1.0 / 298.257223563         # 地球扁率

    # ==========================================
    # 1. 基础转换: 大地坐标 (LLA) <--> 地固系 (ECEF)
    # ==========================================
    @classmethod
    def pos2ecef(cls, pos: np.ndarray, rad = 'rad') -> np.ndarray:
        """
        大地坐标转 ECEF 坐标
        :param pos: [纬度(rad), 经度(rad), 高程(m)]
        :param rad: 纬度和经度的单位，'rad' 表示弧度，'deg' 表示角度
        :return: [X, Y, Z] (m)
        """
        lat, lon, h = pos
        if rad == 'deg':
            lat = math.radians(lat)
            lon = math.radians(lon)
        sinp, cosp = math.sin(lat), math.cos(lat)
        sinl, cosl = math.sin(lon), math.cos(lon)
        
        e2 = cls.FE * (2.0 - cls.FE)
        v = cls.RE / math.sqrt(1.0 - e2 * sinp**2)
        
        x = (v + h) * cosp * cosl
        y = (v + h) * cosp * sinl
        z = (v * (1.0 - e2) + h) * sinp
        return np.array([x, y, z])

    @classmethod
    def ecef2pos(cls, r: np.ndarray) -> np.ndarray:
        """
        ECEF 坐标转大地坐标
        :param r: [X, Y, Z] (m)
        :return: [纬度(rad), 经度(rad), 高程(m)]
        """
        x, y, z = r
        e2 = cls.FE * (2.0 - cls.FE)
        r2 = x**2 + y**2
        
        v = cls.RE
        z_temp, zk = z, 0.0
        
        while abs(z_temp - zk) >= 1E-4:
            zk = z_temp
            sinp = z_temp / math.sqrt(r2 + z_temp**2)
            v = cls.RE / math.sqrt(1.0 - e2 * sinp**2)
            z_temp = z + v * e2 * sinp
            
        if r2 > 1E-12:
            lat = math.atan(z_temp / math.sqrt(r2))
            lon = math.atan2(y, x)
        else:
            lat = math.pi/2.0 if z > 0 else -math.pi/2.0
            lon = 0.0
            
        h = math.sqrt(r2 + z_temp**2) - v
        return np.array([lat, lon, h])

    # ==========================================
    # 2. 站心坐标转换: ENU <--> ECEF
    # ==========================================
    @classmethod
    def _xyz2enu_matrix(cls, pos: np.ndarray) -> np.ndarray:
        """内部方法：计算 ECEF 到 ENU 的旋转矩阵"""
        lat, lon = pos[0], pos[1]
        sinp, cosp = math.sin(lat), math.cos(lat)
        sinl, cosl = math.sin(lon), math.cos(lon)
        return np.array([
            [-sinl,         cosl,        0.0 ],
            [-sinp * cosl, -sinp * sinl, cosp],
            [ cosp * cosl,  cosp * sinl, sinp]
        ])

    @classmethod
    def ecef2enu(cls, pos: np.ndarray, r_ecef: np.ndarray) -> np.ndarray:
        """
        ECEF 向量转 ENU (地方坐标系)
        :param pos: 测站的大地坐标 [lat(rad), lon(rad), h(m)]
        :param r_ecef: 待转换的 ECEF 向量 [x, y, z]
        :return: [E, N, U]
        """
        E = cls._xyz2enu_matrix(pos)
        return E @ r_ecef

    @classmethod
    def enu2ecef(cls, pos: np.ndarray, r_enu: np.ndarray) -> np.ndarray:
        """
        ENU 向量转 ECEF
        :param pos: 测站的大地坐标 [lat(rad), lon(rad), h(m)]
        :param r_enu: 待转换的 ENU 向量 [E, N, U]
        :return: [x, y, z]
        """
        E = cls._xyz2enu_matrix(pos)
        return E.T @ r_enu

    # ==========================================
    # 3. 惯性系转换 (基于 Astropy): ECI <--> ECEF
    # ==========================================
    @classmethod
    def eci2ecef(cls, utc_time_str: str, eci_pos: np.ndarray) -> np.ndarray:
        """
        ECI (地心惯性系 GCRS) 转 ECEF (地心地固系 ITRS)
        :param utc_time_str: UTC时间字符串，如 "2024-01-01T12:00:00"
        :param eci_pos: ECI 下的坐标 [X, Y, Z] (m)
        :return: ECEF 下的坐标 [X, Y, Z] (m)
        """
        t = Time(utc_time_str, scale='utc')
        # 构建 ECI 坐标对象
        eci_coord = GCRS(
            x=eci_pos[0] * u.m, 
            y=eci_pos[1] * u.m, 
            z=eci_pos[2] * u.m, 
            representation_type=CartesianRepresentation, 
            obstime=t
        )
        # 转换到 ECEF
        ecef_coord = eci_coord.transform_to(ITRS(obstime=t))
        return np.array([ecef_coord.x.value, ecef_coord.y.value, ecef_coord.z.value])

    @classmethod
    def ecef2eci(cls, utc_time_str: str, ecef_pos: np.ndarray) -> np.ndarray:
        """
        ECEF (地心地固系 ITRS) 转 ECI (地心惯性系 GCRS)
        :param utc_time_str: UTC时间字符串
        :param ecef_pos: ECEF 下的坐标 [X, Y, Z] (m)
        :return: ECI 下的坐标 [X, Y, Z] (m)
        """
        t = Time(utc_time_str, scale='tai')
        # 构建 ECEF 坐标对象
        ecef_coord = ITRS(
            x=ecef_pos[0] * u.m, 
            y=ecef_pos[1] * u.m, 
            z=ecef_pos[2] * u.m, 
            representation_type=CartesianRepresentation, 
            obstime=t
        )
        # 转换到 ECI
        eci_coord = ecef_coord.transform_to(GCRS(obstime=t))
        return np.array([
            eci_coord.cartesian.x.to_value(u.m), 
            eci_coord.cartesian.y.to_value(u.m), 
            eci_coord.cartesian.z.to_value(u.m)
        ])
    
    @staticmethod
    def get_sun_ecef_astropy(utc_time_str: str) -> np.ndarray:
        """
        使用 astropy 获取太阳在 ECEF 坐标系下的精确三维位置
        :param utc_time_str: UTC时间字符串，如 "2024-01-01T12:00:00"
        :return: 太阳在 ECEF 下的坐标 [X, Y, Z] (单位：米)
        """
        # 1. 统一时间格式
        t = Time(utc_time_str, scale='tai')
        
        # 2. 获取太阳在地心惯性系（GCRS）下的坐标
        sun_gcrs = get_sun(t)
        
        # 3. 转换到地心地固系（ITRS，即 ECEF）
        sun_itrs = sun_gcrs.transform_to(ITRS(obstime=t))
        
        # 4. 提取笛卡尔数值 (注意加上 .cartesian 避免报错)
        return np.array([
            sun_itrs.cartesian.x.to_value(u.m),
            sun_itrs.cartesian.y.to_value(u.m),
            sun_itrs.cartesian.z.to_value(u.m)
        ])
    
    @staticmethod
    def ecef2azel(station_ecef: np.ndarray, sat_ecef: np.ndarray, angle_unit: str) -> tuple:
        """
        计算卫星的方位角和仰角
        :param station_ecef: 测站在 ECEF 坐标系下的坐标 [X, Y, Z] (m)
        :param sat_ecef: 卫星在 ECEF 坐标系下的坐标 [X, Y, Z] (m)
        :param angle_unit: 角度单位，'deg' 表示度，'rad' 表示弧度
        :return: (azimuth, elevation) 单位：度
        """
        # 1. 计算测站的地理坐标（纬度、经度、高程）
        station_lla = GNSSCoordinate.ecef2pos(station_ecef)
        
        # 2. 计算从测站到卫星的视线向量
        los_ecef = sat_ecef - station_ecef
        
        # 3. 将视线向量转换到 ENU 坐标系
        los_enu = GNSSCoordinate.ecef2enu(station_lla, los_ecef)
        
        # 4. 计算方位角和仰角
        east, north, up = los_enu
        azimuth = math.atan2(east, north) * 180.0 / math.pi
        elevation = math.atan2(up, math.sqrt(east**2 + north**2)) * 180.0 / math.pi
        
        # 确保方位角在 [0, 360) 范围内
        if azimuth < 0:
            azimuth += 360.0
        
        if angle_unit == 'rad':
            azimuth = math.radians(azimuth)
            elevation = math.radians(elevation)

        return azimuth, elevation


if __name__ == "__main__":
    import numpy as np
    from Coord import GNSSCoordinate as GC

    # 1. 模拟一个测站坐标 (注意：lat和lon必须是弧度)
    lat_rad = np.radians(22.32) # 香港纬度
    lon_rad = np.radians(114.17) # 香港经度
    h = 10.0
    pos_lla = np.array([lat_rad, lon_rad, h])

    # LLA 转 ECEF
    station_ecef = GC.pos2ecef(pos_lla)
    print(f"测站 ECEF: {station_ecef} 米")

    # 2. 模拟一个从精密星历 (SP3) 读到的卫星向量
    sat_ecef = np.array([15600000.0, 15600000.0, 15600000.0])
    # 计算从测站到卫星的视线向量 (ECEF)
    los_ecef = sat_ecef - station_ecef

    # 将视线向量转为 ENU (方便计算高度角、方位角)
    los_enu = GC.ecef2enu(pos_lla, los_ecef)
    print(f"视线向量 ENU: {los_enu} 米")

    # 3. 卫星轨道积分需求：ECI 转 ECEF
    obs_time = "2024-05-20T08:00:00"
    sat_eci = np.array([20000000.0, 10000000.0, 5000000.0]) # 模拟惯性系坐标

    sat_ecef_converted = GC.eci2ecef(obs_time, sat_eci)
    print(f"卫星 ECEF: {sat_ecef_converted} 米")

    eci_converted_back = GC.ecef2eci(obs_time, sat_ecef_converted)
    print(f"转换回 ECI: {eci_converted_back} 米")

    print(f"太阳 ECEF: {GC.get_sun_ecef_astropy(obs_time)} 米")