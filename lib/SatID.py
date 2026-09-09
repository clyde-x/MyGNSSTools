
class SatID:
    # 1. 严格对应 C 代码中的宏定义 (系统标识符, MIN_PRN, MAX_PRN)
    # 按照 RTKLIB 计算 sat 编号时的累加顺序排列
    _SYS_MACROS = [
        ('G', 1, 32),    # GPS (NSATGPS = 32)
        ('R', 1, 27),    # GLONASS (NSATGLO = 27)
        ('E', 1, 36),    # Galileo (NSATGAL = 36)
        ('J', 193, 202), # QZSS (NSATQZS = 10)
        ('C', 1, 60),    # BeiDou (NSATCMP = 46)  ??60
        ('I', 1, 14),    # IRNSS (NSATIRN = 14)
        ('L', 1, 10),    # LEO (NSATLEO = 10)
        ('S', 120, 158)  # SBAS (NSATSBS = 39)
    ]

    # 2. 动态生成配置字典：'G': {'min': 1, 'max': 32, 'offset': 0}
    SYS_CONFIG = {}
    _current_offset = 0
    for _sys_char, _min_prn, _max_prn in _SYS_MACROS:
        _nsat = _max_prn - _min_prn + 1
        SYS_CONFIG[_sys_char] = {
            'min': _min_prn,
            'max': _max_prn,
            'offset': _current_offset
        }
        _current_offset += _nsat
        
    MAX_RTK_SAT = _current_offset # 总卫星数 (根据你提供的宏应为 214)

    def __init__(self, sat_input):
        self.sys = None    # 卫星系统字符，例如 'G', 'C', 'J'
        self.prn = 0       # 实际的物理 PRN 编号，例如 1, 193
        self.rtk_sat = 0   # RTKLIB 中的连续整数编号，例如 1, 96

        if isinstance(sat_input, str):
            self._init_from_str(sat_input)
        elif isinstance(sat_input, int):
            self._init_from_int(sat_input)
        else:
            raise TypeError("初始化参数必须是字符串(如'G01')或整数(RTKLIB编号)")

    def _init_from_str(self, prn_str):
        """解析字符串，支持标准 RINEX 缩写 (如 J01) 或实际 PRN (如 J193)"""
        prn_str = prn_str.strip().upper()
        if len(prn_str) < 2:
            raise ValueError(f"无效的卫星字符串: {prn_str}")

        sys_char = prn_str[0]
        if sys_char not in self.SYS_CONFIG:
            raise ValueError(f"未知的卫星系统标识: {sys_char}")

        prn_val = int(prn_str[1:])
        config = self.SYS_CONFIG[sys_char]

        # 智能适配 RINEX 标准缩写
        # QZSS: J01~J10 实际对应 PRN 193~202
        if sys_char == 'J' and 1 <= prn_val <= 10:
            prn_val += 192 
        # SBAS: S20~S58 实际对应 PRN 120~158 (也有直接写S120的)
        elif sys_char == 'S' and 20 <= prn_val <= 58:
            prn_val += 100

        # 校验范围
        if not (config['min'] <= prn_val <= config['max']):
            raise ValueError(f"PRN超出范围: {sys_char} 系统应在 {config['min']}-{config['max']} 之间")

        self.sys = sys_char
        self.prn = prn_val
        # RTKLIB 计算公式: prn - MINPRN + 1 + offset
        # 为了符合编程中从0开始的逻辑，化简为: prn - min + offset + 1
        self.rtk_sat = self.prn - config['min'] + 1 + config['offset']

    def _init_from_int(self, rtk_sat):
        """解析 RTKLIB 内部连续整数编号"""
        if not (1 <= rtk_sat <= self.MAX_RTK_SAT):
            raise ValueError(f"无效的RTKLIB卫星编号: {rtk_sat} (超出最大范围 {self.MAX_RTK_SAT})")

        self.rtk_sat = rtk_sat
        
        # 逆向查找: 倒序遍历配置字典，找到属于哪个区间
        # Python 3.7+ 字典保持插入顺序
        for sys_char in reversed(list(self.SYS_CONFIG.keys())):
            config = self.SYS_CONFIG[sys_char]
            if rtk_sat > config['offset']:
                self.sys = sys_char
                # 逆向公式: prn = rtk_sat - offset - 1 + min
                self.prn = rtk_sat - config['offset'] - 1 + config['min']
                break

    @property
    def prn_str(self):
        """输出 RINEX 标准的字符串格式"""
        if self.sys == 'J':
            # QZSS 习惯输出 J01, 隐去 193 的物理 PRN
            return f"J{self.prn - 192:02d}"
        elif self.sys == 'S':
            # SBAS 习惯输出 S20
            return f"S{self.prn - 100:02d}"
        else:
            return f"{self.sys}{self.prn:02d}"
    
    def csu_prn(self):
        if self.sys == 'G':
            return int(f"{self.prn:02d}")
        if self.sys == 'C':
            return int(f"1{self.prn:02d}")
        if self.sys == 'R':
            return int(f"2{self.prn:02d}")
        if self.sys == 'E':
            return int(f"3{self.prn:02d}")
        if self.sys == 'J':
            return int(f"4{self.prn:02d}")
        else:
            raise ValueError(f"不支持的卫星系统 {self.sys}，无法转换为 CSU PRN")

    def __str__(self):
        return self.prn_str

    def __repr__(self):
        return f"SatID('{self.prn_str}', prn={self.prn}, rtk={self.rtk_sat})"

    def __eq__(self, other):
        if isinstance(other, SatID):
            return self.rtk_sat == other.rtk_sat
        elif isinstance(other, str):
            return self.prn_str == other
        return False
    
    def __hash__(self):
        return hash(self.rtk_sat)


def sort(in_ls, ifstr=True):
    prn_ls = []
    for prn in in_ls:
        if isinstance(prn, str):
            prn_ls.append(SatID(prn))
        elif isinstance(prn, SatID):
            prn_ls.append(prn)
        else:
            raise TypeError(f"输入列表中的元素必须是字符串或 SatID 对象，当前类型: {type(prn)}")
    prn_ls.sort(key=lambda x: x.rtk_sat)
    if ifstr:
        return [sat.prn_str for sat in prn_ls]
    else:
        return prn_ls
    
def filter(in_ls, sys_list=None, exclude_prn=None):
    if sys_list is None:
        sys_list = ['G', 'C', 'R', 'E', 'J', 'S']
    if exclude_prn is None:
        exclude_prn = []
    filtered_sats = []
    for prn in in_ls:
        if isinstance(prn, str):
            if prn[0] in sys_list and prn not in exclude_prn:
                filtered_sats.append(prn)
        elif isinstance(prn, SatID):
            if prn.sys in sys_list and prn.prn not in exclude_prn:
                filtered_sats.append(prn)
    return filtered_sats