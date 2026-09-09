'''
parse_file(file) -> List[GNSSBaseState]
GNSSBaseState : PosState | VelAccState | IonoState | TropState | SatState
GNSSBaseState : type, week, tow, timesystem
'''

import sys
import os
sys.path.append("../lib")

import TimeSystem
import SatID 
from dataclasses import dataclass

# ==========================================
# 定义父类（基类）
# ==========================================
@dataclass
class GNSSBaseState:
    """所有 GNSS 状态记录的基类"""
    type: str               # 记录类型 POS, VELACC, ION, TROP, SAT
    week: int               # GPS周号
    tow: float              # 周内时间（秒）
    timesystem: TimeSystem  # 解析后生成的 TimeSystem 对象

# ==========================================
# 子类定义（继承自 GNSSBaseState）
# ==========================================
@dataclass
class PosState(GNSSBaseState):
    """记录流动站位置状态 ($POS)"""
    stat: int       # 解算状态
    posx: float     # ECEF X坐标 (浮点解)
    posy: float     # ECEF Y坐标 (浮点解)
    posz: float     # ECEF Z坐标 (浮点解)
    posxf: float    # ECEF X坐标 (固定解)
    posyf: float    # ECEF Y坐标 (固定解)
    poszf: float    # ECEF Z坐标 (固定解)

    @classmethod
    def from_string(cls, line: str):
        parts = line.strip().split(',')
        if parts[0] != '$POS' or len(parts) != 10:
            raise ValueError(f"无效的 $POS 记录: {line}")
        
        return cls(
            type="POS",  # 显式传入固定类型
            week=int(parts[1]), 
            tow=float(parts[2]),
            timesystem=TimeSystem.TimeSystem().from_gps(week=int(parts[1]), sow=float(parts[2])),
            stat=int(parts[3]),
            posx=float(parts[4]), posy=float(parts[5]), posz=float(parts[6]),
            posxf=float(parts[7]), posyf=float(parts[8]), poszf=float(parts[9])
        )

    def __repr__(self):
        return (f"PosState(time={self.timesystem}, stat={self.stat}, "
                f"posx={self.posx}, posy={self.posy}, posz={self.posz}, "
                f"posxf={self.posxf}, posyf={self.posyf}, poszf={self.poszf})")

@dataclass
class VelAccState(GNSSBaseState):
    """记录流动站速度和加速度状态 ($VELACC)"""
    stat: int       # 解算状态
    vele: float     # 速度东向 (浮点解)
    veln: float     # 速度北向 (浮点解)
    velu: float     # 速度天向 (浮点解)
    acce: float     # 加速度东向 (浮点解)
    accn: float     # 加速度北向 (浮点解)
    accu: float     # 加速度天向 (浮点解)
    velef: float    # 速度东向 (固定解)
    velnf: float    # 速度北向 (固定解)
    veluf: float    # 速度天向 (固定解)
    accef: float    # 加速度东向 (固定解)
    accnf: float    # 加速度北向 (固定解)
    accuf: float    # 加速度天向 (固定解)

    @classmethod
    def from_string(cls, line: str):
        parts = line.strip().split(',')
        if parts[0] != '$VELACC' or len(parts) != 16:
            raise ValueError(f"无效的 $VELACC 记录: {line}")
        
        return cls(
            type="VELACC",
            week=int(parts[1]), 
            tow=float(parts[2]),
            timesystem=TimeSystem.TimeSystem().from_gps(week=int(parts[1]), sow=float(parts[2])),
            stat=int(parts[3]),
            vele=float(parts[4]), veln=float(parts[5]), velu=float(parts[6]),
            acce=float(parts[7]), accn=float(parts[8]), accu=float(parts[9]),
            velef=float(parts[10]), velnf=float(parts[11]), veluf=float(parts[12]),
            accef=float(parts[13]), accnf=float(parts[14]), accuf=float(parts[15])
        )
    
    def __repr__(self):
        return (f"VelAccState(time={self.timesystem}, stat={self.stat}, "
                f"vele={self.vele}, veln={self.veln}, velu={self.velu}, "
                f"acce={self.acce}, accn={self.accn}, accu={self.accu}, "
                f"velef={self.velef}, velnf={self.velnf}, veluf={self.veluf}, "
                f'accef={self.accef}, accnf={self.accnf}, accuf={self.accuf})')

@dataclass
class IonoState(GNSSBaseState):
    """记录电离层参数状态 ($ION)"""
    stat: int       # 解算状态
    sat_id: str     # 卫星ID
    az: float       # 方位角（度）
    el: float       # 仰角（度）
    ion: float      # L1电离层延迟 (m, 浮点解)
    ion_fixed: float # L1电离层延迟 (m, 固定解)

    @classmethod
    def from_string(cls, line: str):
        parts = line.strip().split(',')
        if parts[0] != '$ION' or len(parts) != 9:
            raise ValueError(f"无效的 $ION 记录: {line}")
        
        return cls(
            type="ION",
            week=int(parts[1]), 
            tow=float(parts[2]),
            timesystem=TimeSystem.TimeSystem().from_gps(week=int(parts[1]), sow=float(parts[2])),
            stat=int(parts[3]),
            sat_id=parts[4], az=float(parts[5]), el=float(parts[6]),
            ion=float(parts[7]), ion_fixed=float(parts[8])
        )

    def __repr__(self):
        return (f"IonoState(time={self.timesystem}, stat={self.stat}, "
                f"sat_id='{self.sat_id}', az={self.az}, el={self.el}, "
                f"ion={self.ion}, ion_fixed={self.ion_fixed})")
    
@dataclass
class TropState(GNSSBaseState):
    """记录对流层参数状态 ($TROP)"""
    stat: int       # 解算状态
    rcv: int        # 接收机（1:流动站，2:基准站）
    ztd: float      # 顶点总延迟 (m, 浮点解)
    ztdf: float     # 顶点总延迟 (m, 固定解)

    @classmethod
    def from_string(cls, line: str):
        parts = line.strip().split(',')
        if parts[0] != '$TROP' or len(parts) != 7:
            raise ValueError(f"无效的 $TROP 记录: {line}")
        
        return cls(
            type="TROP",
            week=int(parts[1]), 
            tow=float(parts[2]),
            timesystem=TimeSystem.TimeSystem().from_gps(week=int(parts[1]), sow=float(parts[2])),
            stat=int(parts[3]),
            rcv=int(parts[4]), ztd=float(parts[5]), ztdf=float(parts[6])
        )

    def __repr__(self):
        return (f"TropState(time={self.timesystem}, stat={self.stat}, "
                f"rcv={self.rcv}, ztd={self.ztd}, ztdf={self.ztdf})")

@dataclass
class SatState(GNSSBaseState):
    """记录伪距和载波相位观测值的残差及状态 ($SAT)"""
    sat_id: str       # 卫星ID (这里保留了你的类型声明)
    frq: int          # 频率索引（1:L1, 2:L2, 3:L5...）
    az: float         # 方位角（度）
    el: float         # 仰角（度）
    resp: float       # 伪距残差（m）
    resc: float       # 载波相位残差（m）
    vsat: int         # 有效数据标志（0:无效，1:有效）
    snr: float        # 信号强度（dbHz）
    fix: int          # 模糊度标志（0:无, 1:非AR, 2:AR集合, 3:保持集合）
    slip: int         # 周跳标志
    lock: int         # 载波锁定计数
    outc: int         # 数据中断计数
    slipc: int        # 周跳计数
    rejc: int         # 数据拒绝（异常值）计数
    icbias: float     # 通道间偏差（GLONASS专用）
    bias: float       # 相位偏差
    bias_var: float   # 相位偏差的方差
    wavelength: float # 波长 

    @classmethod
    def from_string(cls, line: str):
        parts = line.strip().split(',')
        satid = SatID.SatID(parts[3])

        # 提取共用的基类时间属性
        week_val = int(parts[1])
        tow_val = float(parts[2])
        timesys_val = TimeSystem.TimeSystem().from_gps(week=week_val, sow=tow_val)

        if satid.sys == 'R':
            if parts[0] != '$SAT' or len(parts) != 21:
                raise ValueError(f"无效的 $SAT 记录 (预期21个字段，实际{len(parts)}个): {line}")
            
            return cls(
                type="SAT", week=week_val, tow=tow_val, timesystem=timesys_val,
                sat_id=satid, frq=int(parts[4]),
                az=float(parts[5]), el=float(parts[6]),
                resp=float(parts[7]), resc=float(parts[8]),
                vsat=int(parts[9]), snr=float(parts[10]),
                fix=int(parts[11]), slip=int(parts[12]),
                lock=int(parts[13]), outc=int(parts[14]),
                slipc=int(parts[15]), rejc=int(parts[16]),
                icbias=float(parts[17]), bias=float(parts[18]),
                bias_var=float(parts[19]), wavelength=float(parts[20])
            )
        else:
            if parts[0] != '$SAT' or len(parts) != 20:
                raise ValueError(f"无效的 $SAT 记录 (预期20个字段，实际{len(parts)}个): {line}")
            
            return cls(
                type="SAT", week=week_val, tow=tow_val, timesystem=timesys_val,
                sat_id=satid, frq=int(parts[4]),
                az=float(parts[5]), el=float(parts[6]),
                resp=float(parts[7]), resc=float(parts[8]),
                vsat=int(parts[9]), snr=float(parts[10]),
                fix=int(parts[11]), slip=int(parts[12]),
                lock=int(parts[13]), outc=int(parts[14]),
                slipc=int(parts[15]), rejc=int(parts[16]),
                icbias=0.0, bias=float(parts[17]),
                bias_var=float(parts[18]), wavelength=float(parts[19])
            )

    def __repr__(self):        
        return (f"SatState(time={self.timesystem}, sat_id='{self.sat_id}', frq={self.frq}, "
                f"az={self.az}, el={self.el}, resp={self.resp}, resc={self.resc}, vsat={self.vsat}, "
                f"snr={self.snr}, fix={self.fix}, slip={self.slip}, lock={self.lock}, outc={self.outc}, "
                f"slipc={self.slipc}, rejc={self.rejc}, icbias={self.icbias}, bias={self.bias}, "
                f"bias_var={self.bias_var}, wavelength={self.wavelength})")

# ==========================================
# 解析函数
# ==========================================
def parse_line(line: str):
    if line.startswith('$POS'):
        return PosState.from_string(line)
    elif line.startswith('$VELACC'):
        return VelAccState.from_string(line)
    elif line.startswith('$ION'):
        return IonoState.from_string(line)
    elif line.startswith('$TROP'):
        return TropState.from_string(line)
    elif line.startswith('$SAT'):
        return SatState.from_string(line)
    else:
        return None
    
def parse_file(file: str):
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    records = []
    for line in lines:
        record = parse_line(line)
        if record is not None:
            records.append(record)
    return records