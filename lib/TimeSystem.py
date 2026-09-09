"""
2026-03       v 0.2
定义了TimeSystem类，用于处理时间的各种表示形式。
- 添加了 GPS Week 和 GPS SOW (周内秒) 的支持及初始化方法。
- 修复了 from_ymdhms 中小数秒精度丢失的问题，采用 timedelta 彻底解决进位溢出。
- 实现了 from_mjd 方法。
- 重载了运算、比较以及格式化输出等方法。

没有UTC跳秒
"""

import time
import datetime
import re

class TimeSystem:
    # 预定义常量
    GPS_EPOCH = datetime.datetime(1980, 1, 6, 0, 0, 0)
    MJD_EPOCH = datetime.datetime(1858, 11, 17, 0, 0, 0)

    def __init__(self):
        self._reset()

    def _reset(self):
        """重置所有内部属性"""
        self.year = None
        self.month = None
        self.day = None
        self.hour = None
        self.minute = None
        self.second = None  # float（支持小数秒）
        self.mjd = None
        self.mjd_int = None
        self.mjd_frac = None
        self.gps_week = None
        self.gps_sow = None
        self.datetime = None
        self.timestamp = None
    
    def from_datetime(self, dt: datetime.datetime):
        """从 datetime 对象初始化"""
        return self._update_from_datetime(dt)

    def _update_from_datetime(self, dt: datetime.datetime):
        """核心方法：从 datetime 对象统一更新所有属性"""
        if not isinstance(dt, datetime.datetime):
            raise TypeError("Expected datetime.datetime instance")

        # 保存 datetime 并提取常规时间字段
        self.datetime = dt
        self.year = dt.year
        self.month = dt.month
        self.day = dt.day
        self.hour = dt.hour
        self.minute = dt.minute
        self.second = dt.second + dt.microsecond / 1_000_000.0  # 保留小数秒

        # 计算 MJD
        self.mjd, self.mjd_int, self.mjd_frac = self._datetime_to_mjd(dt)

        # 计算 GPS Week & SOW (假设传入的时间与GPS时间基准一致，未引入跳秒查找表)
        delta_gps = dt - self.GPS_EPOCH
        total_gps_seconds = delta_gps.total_seconds()
        self.gps_week = int(total_gps_seconds // 604800)
        self.gps_sow = total_gps_seconds % 604800

        # 计算 timestamp（本地时区）
        self.timestamp = time.mktime(dt.timetuple()) + dt.microsecond / 1_000_000.0

        return self  # 支持链式调用

    @staticmethod
    def _datetime_to_mjd(dt: datetime.datetime):
        """将 datetime 转换为 MJD（Modified Julian Date）"""
        year, month, day = dt.year, dt.month, dt.day
        hour, minute = dt.hour, dt.minute
        second = dt.second + dt.microsecond / 1_000_000.0

        fractional_day = (hour + minute / 60.0 + second / 3600.0) / 24.0

        if month <= 2:
            year -= 1
            month += 12

        a = year // 100
        b = 2 - a + a // 4
        jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5
        mjd = jd - 2400000.5 + fractional_day

        mjd_int = int(mjd)
        mjd_frac = mjd - mjd_int
        return mjd, mjd_int, mjd_frac

    # ==========================
    # 公共初始化接口（返回 self）
    # ==========================

    def from_ymdhms(self, year, month, day, hour, minute, second):
        """从年月日时分秒初始化，利用 timedelta 完美规避进位与小数精度问题"""
        try:
            # 先创建一个不含时分秒的基准日
            base_dt = datetime.datetime(int(year), int(month), int(day))
            # 所有的溢出（如 second=60.5, minute=60）都交给 timedelta 自动处理
            delta = datetime.timedelta(hours=int(hour), minutes=int(minute), seconds=float(second))
            return self._update_from_datetime(base_dt + delta)
        except ValueError as e:
            raise ValueError(f"Invalid date/time components provided.") from e

    def from_gps(self, week: int, sow: float):
        """从 GPS 周 和 周内秒 初始化"""
        dt = self.GPS_EPOCH + datetime.timedelta(weeks=int(week), seconds=float(sow))
        return self._update_from_datetime(dt)

    def from_str(self, timestr: str):
        """
        从字符串初始化，格式如：
        'YYYY MM DD HH mm SS.ssssss' 或 'YYYY-MM-DD HH:mm:SS.ssssss' 或 ISO 'YYYY-MM-DDTHH:mm:SS'
        """
        # 标准化分隔符（包括 ISO 格式的 'T'）为空格
        normalized = re.sub(r'[-:T/]', ' ', timestr.strip())
        parts = normalized.split()
        if len(parts) < 6:
            raise ValueError(f"Time string must contain at least 6 components: {timestr}")

        try:
            return self.from_ymdhms(parts[0], parts[1], parts[2], parts[3], parts[4], parts[5])
        except Exception as e:
            raise ValueError(f"Invalid time string format: {timestr}") from e

    def from_mjd(self, mjd: float):
        """从 MJD 恢复时间"""
        dt = self.MJD_EPOCH + datetime.timedelta(days=float(mjd))
        return self._update_from_datetime(dt)

    def current_time(self):
        """设置为当前本地时间"""
        return self._update_from_datetime(datetime.datetime.now())

    def get_ymdhms(self):
        """返回 (year, month, day, hour, minute, second)"""
        return self.year, self.month, self.day, self.hour, self.minute, self.second

    # ==========================
    # 运算符重载
    # ==========================

    def __lt__(self, other):
        if not isinstance(other, TimeSystem): return NotImplemented
        return self.datetime < other.datetime

    def __le__(self, other):
        if not isinstance(other, TimeSystem): return NotImplemented
        return self.datetime <= other.datetime

    def __eq__(self, other):
        if not isinstance(other, TimeSystem): return NotImplemented
        return self.datetime == other.datetime

    def __hash__(self):
        return hash(self.datetime)

    def __gt__(self, other):
        if not isinstance(other, TimeSystem): return NotImplemented
        return self.datetime > other.datetime

    def __ge__(self, other):
        if not isinstance(other, TimeSystem): return NotImplemented
        return self.datetime >= other.datetime

    def __sub__(self, other):
        """返回两时间差（秒）"""
        if not isinstance(other, TimeSystem):
            raise TypeError("Can only subtract another TimeSystem instance")
        return (self.datetime - other.datetime).total_seconds()

    def __add__(self, seconds):
        """返回增加秒数后的新 TimeSystem 对象"""
        if not isinstance(seconds, (int, float)):
            raise TypeError("Can only add seconds (int or float)")
        new_dt = self.datetime + datetime.timedelta(seconds=seconds)
        return TimeSystem()._update_from_datetime(new_dt)

    # ==========================
    # 字符串与格式化
    # ==========================

    def __str__(self):
        if self.datetime is None:
            return "<Uninitialized TimeSystem>"
        # 如果需要打印出带小数的秒，可以改写此处
        return self.datetime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] # 保留到毫秒

    def to_str(self, fmt='%Y-%m-%d %H:%M:%S.%f'):
        return self.datetime.strftime(fmt) if self.datetime else ""

    def to_digital(self):
        if self.datetime is None:
            return ""
        return self.datetime.strftime('%Y%m%d%H%M%S')


# ==========================
# 针对 DataFrame 操作的辅助函数
# ==========================

def make_ts(row):
    """
    Usage: df['timesystem'] = df.apply(make_ts, axis=1)
    """
    return TimeSystem().from_ymdhms(
        row['year'], row['month'], row['day'],
        row['hour'], row['minute'], row['second']
    )

def common_ts(ts1, ts2):
    """
    返回两个 TimeSystem 列表/可迭代对象的共有 TimeSystem 列表（基于 to_digital() 比较）。
    """
    keys2 = {t.to_digital() for t in ts2 if t is not None}
    seen = set()
    common = []
    for t in ts1:
        if t is None:
            continue
        k = t.to_digital()
        if k in keys2 and k not in seen:
            common.append(t)
            seen.add(k)
    return common

def align_dataframes_on_times(df1, df2, name1='timesystem', name2='timesystem'):
    """
    返回只保留两表共有时间条目的副本（基于 timesystem.to_digital() 比较）。
    """
    s1 = df1[name1].apply(lambda t: t.to_digital())
    s2 = df2[name2].apply(lambda t: t.to_digital())
    common_keys = set(s1) & set(s2)

    df1_common = df1[s1.isin(common_keys)].reset_index(drop=True)
    df2_common = df2[s2.isin(common_keys)].reset_index(drop=True)
    print(f"Common time entries: {len(common_keys)}, df1: {len(df1_common)}, df2: {len(df2_common)}")
    return df1_common, df2_common

def find_interval_indices(ts_list, target_ts):
    """
    在 ts_list 中找到 target_ts 所在的区间 [ts_list[i], ts_list[i+1]) 的索引 i。
    返回 (i, i+1) 或 (None, None) 如果 target_ts 不在范围内。
    """
    for i in range(len(ts_list) - 1):
        if ts_list[i] <= target_ts < ts_list[i + 1]:
            return i, ts_list[i], i+1, ts_list[i + 1]
    return None, None, None, None