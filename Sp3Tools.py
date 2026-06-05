"""
2026-03       v 0.3
定义了SP3类，用于处理SP3文件 (支持 SP3-c/d 格式)
- 全面引入 TimeSystem 智能时间对象，接管所有时间解析与转换。
- 修复了 SP3 文件头 (17颗卫星换行) 的规范写入。
- 自动生成 gps_time (连续秒)，支持与其他高频数据（如 KBR 观测值）进行无缝时间对齐。
- 支持位置、速度及其标准差 (Sigma) 的完整解析。
"""

import os
import warnings
import pandas as pd
import TimeSystem

class SP3:
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.header = {}
        self.data = []  # 存储按历元组织的嵌套字典
        self.df = pd.DataFrame() 
        self.sat_id_list = []
        self.sat_acc_list = []
        
        # 标志位：数据中是否包含速度或偏差
        self.withVelocity = False
        self.withDeviation = False
        
        if self.filepath and os.path.exists(self.filepath):
            self.parse()

    def parse(self):
        with open(self.filepath, 'r') as file:
            lines = file.readlines()
        
        header_lines = []
        data_lines = []
        in_header = True
        for line in lines:
            if in_header:
                if line.startswith('*'):
                    data_lines.append(line)
                    in_header = False
                else:
                    header_lines.append(line)
            else:
                data_lines.append(line)
        
        self._parse_header(header_lines)
        self._parse_data(data_lines)

    def _parse_float(self, string_val, default=0.0):
        """安全地将字符串转为浮点数，处理 SP3 中的空缺值"""
        s = string_val.strip()
        if not s: return default
        try: return float(s)
        except ValueError: return default

    def _parse_header(self, header_lines):
        line_index = 0
        self.header['comments'] = ''
        for line in header_lines:
            line_index += 1
            if line.startswith('##'):
                parts = line.split()
                self.header['gps_week'] = int(parts[1])
                self.header['gps_seconds'] = float(parts[2])
                self.header['interval'] = float(parts[3])
                self.header['mjd'] = int(parts[4])
                self.header['mjd_seconds'] = float(parts[5])

            elif line.startswith('#'):
                # 第一行: #cP YYYY MM DD HH mm SS.SSSSSSS ...
                self.header['version'] = line[1] # 'c' or 'd'
                self.header['velocity'] = line[2] # 'V' or 'P'
                self.withVelocity = (self.header['velocity'] == 'V')
                
                # 【智能时间处理】：直接喂给 TimeSystem
                time_str = line[3:32]
                ts = TimeSystem.TimeSystem().from_str(time_str)
                self.header['time'] = ts
                
                # 同步更新 header 里的零散字段（方便其他地方调用）
                self.header['year'] = ts.year
                self.header['month'] = ts.month
                self.header['day'] = ts.day
                self.header['hour'] = ts.hour
                self.header['minute'] = ts.minute
                self.header['second'] = ts.second
                
                # 解析后续定长字段
                self.header['num_epochs'] = int(line[32:39].strip())
                self.header['used_data'] = line[40:45].strip()
                self.header['coord_system'] = line[46:51].strip()
                self.header['orbit_type'] = line[52:55].strip()
                self.header['agency'] = line[56:60].strip()

            elif line.startswith('+ '): 
                if line_index == 3:
                    self.header['num_sats'] = int(line[3:6].strip())
                # SP3 规定一行最多 17 颗星，占位 9~60
                for i in range(9, min(len(line), 60), 3):
                    sat_id = line[i:i+3].strip()
                    if sat_id and sat_id != '0':
                        self.sat_id_list.append(sat_id)

            elif line.startswith('++'):
                for i in range(9, min(len(line), 60), 3):
                    acc = line[i:i+3].strip()
                    if acc and acc != '0':
                        self.sat_acc_list.append(int(acc))

            elif line.startswith('%c'):
                file_type = line[3:5]
                if file_type.upper() != 'CC':
                    self.header['file_type'] = file_type
                time_system = line[9:12]
                if time_system.upper() != 'CCC':
                    self.header['time_system'] = time_system

            elif line.startswith('%f'):
                parts = line.split()
                if len(parts) > 1:
                    basePV = self._parse_float(parts[1])
                    if basePV > 0: self.header['base_PV'] = basePV
                if len(parts) > 2:
                    baseCLK = self._parse_float(parts[2])
                    if baseCLK > 0: self.header['base_CLK'] = baseCLK

            elif line.startswith('/*'):
                self.header['comments'] += line[2:]

    def _parse_data(self, data_lines):
        epoch = {}
        for line in data_lines:
            if line.upper().startswith('EOF'):
                if epoch:
                    epoch['visible_sats'] = len(epoch['satellites'])
                    self.data.append(epoch)
                break
            
            elif line.startswith('*'):
                if epoch:
                    epoch['visible_sats'] = len(epoch['satellites'])
                    self.data.append(epoch)
                
                # 【智能时间处理】：历元时间解析
                time_str = line[3:32]
                ts = TimeSystem.TimeSystem().from_str(time_str)
                epoch = {'time': ts, 'satellites': {}}

            elif line.startswith('P'):
                sat_id = line[1:4].strip()
                x = self._parse_float(line[4:18])
                y = self._parse_float(line[18:32])
                z = self._parse_float(line[32:46])
                clk = self._parse_float(line[46:60])
                
                # 偏差 (Sigma) 
                sigma_x = self._parse_float(line[61:63]) if len(line) >= 63 else 0
                sigma_y = self._parse_float(line[64:66]) if len(line) >= 66 else 0
                sigma_z = self._parse_float(line[67:69]) if len(line) >= 69 else 0
                sigma_clk = self._parse_float(line[70:73]) if len(line) >= 73 else 0
                
                if sigma_x > 0 or sigma_y > 0 or sigma_z > 0 or sigma_clk > 0:
                    self.withDeviation = True

                epoch['satellites'][sat_id] = {
                    'position': (x, y, z),
                    'clock': clk,
                    'sigma': (sigma_x, sigma_y, sigma_z, sigma_clk)
                }

            elif line.startswith('V'):
                sat_id = line[1:4].strip()
                vx = self._parse_float(line[4:18])
                vy = self._parse_float(line[18:32])
                vz = self._parse_float(line[32:46])
                vclk = self._parse_float(line[46:60])
                
                sigma_vx = self._parse_float(line[61:63]) if len(line) >= 63 else 0
                sigma_vy = self._parse_float(line[64:66]) if len(line) >= 66 else 0
                sigma_vz = self._parse_float(line[67:69]) if len(line) >= 69 else 0
                sigma_vclk = self._parse_float(line[70:73]) if len(line) >= 73 else 0

                if sat_id in epoch['satellites']:
                    epoch['satellites'][sat_id]['velocity'] = (vx, vy, vz)
                    epoch['satellites'][sat_id]['vclk'] = vclk
                    epoch['satellites'][sat_id]['v_sigma'] = (sigma_vx, sigma_vy, sigma_vz, sigma_vclk)

    def write_header(self, output_path):
        with open(output_path, 'w') as file:
            v = self.header.get('version', 'c')
            t = self.header.get('velocity', 'P')
            ts = self.header.get('time')
            yr, mo, dy, hr, mi, sc = ts.get_ymdhms() if ts else (2000,1,1,0,0,0.0)
            
            ep = self.header.get('num_epochs', 0)
            ud = self.header.get('used_data', 'ORBIT').ljust(5)
            cs = self.header.get('coord_system', 'IGS08').ljust(5)
            ot = self.header.get('orbit_type', 'BCE').ljust(3)
            ag = self.header.get('agency', 'IGS').ljust(4)
            
            file.write(f"#{v}{t}{yr:4d} {mo:2d} {dy:2d} {hr:2d} {mi:2d} {sc:11.8f} {ep:7d} {ud} {cs} {ot} {ag}\n")
            
            gw = self.header.get('gps_week', 0)
            gs = self.header.get('gps_seconds', 0.0)
            iv = self.header.get('interval', 0.0)
            md = self.header.get('mjd', 0)
            ms = self.header.get('mjd_seconds', 0.0)
            file.write(f"## {gw:4d} {gs:15.8f} {iv:14.8f} {md:5d} {ms:15.12f}\n")
            
            num_sats = len(self.sat_id_list)
            sat_ids_padded = self.sat_id_list + ['0'] * (85 - num_sats)
            for row in range(5):
                prefix = f"+  {num_sats:3d}   " if row == 0 else "+        "
                line_sats = sat_ids_padded[row*17 : (row+1)*17]
                sat_str = "".join([f"{sid:>3}" if sid != '0' else "  0" for sid in line_sats])
                file.write(prefix + sat_str + "\n")

            acc_padded = self.sat_acc_list + [0] * (85 - len(self.sat_acc_list))
            for row in range(5):
                prefix = "++       "
                line_accs = acc_padded[row*17 : (row+1)*17]
                acc_str = "".join([f"{acc:>3}" for acc in line_accs])
                file.write(prefix + acc_str + "\n")

            file_type = self.header.get('file_type', 'cc').ljust(2)
            time_system = self.header.get('time_system', 'ccc').ljust(3)
            file.write(f"%c {file_type} cc {time_system} ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n")
            file.write(f"%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n")
            
            base_PV = self.header.get('base_PV', 0.0)
            base_CLK = self.header.get('base_CLK', 0.0)
            file.write(f"%f {base_PV:10.7f} {base_CLK:12.9f} {0.0:14.11f} {0.0:18.15f}\n")
            file.write(f"%f {0.0:10.7f} {0.0:12.9f} {0.0:14.11f} {0.0:18.15f}\n")
            file.write("%i    0    0    0    0     0     0     0     0         0\n")
            file.write("%i    0    0    0    0     0     0     0     0         0\n")

            if self.header['comments']:
                comments = self.header['comments'].split('\n')
                for comment in comments:
                    if comment.strip():
                        file.write(f"/*{comment[:58]}\n")

    def write_data(self, output_path):
        with open(output_path, 'a') as file:
            for epoch in self.data:
                ts = epoch['time']
                yr, mo, dy, hr, mi, sc = ts.get_ymdhms()
                file.write(f"*  {yr:4d} {mo:2d} {dy:2d} {hr:2d} {mi:2d} {sc:11.8f}\n")
                
                for sat_id in self.sat_id_list:
                    if sat_id in epoch['satellites']:
                        sat_data = epoch['satellites'][sat_id]
                        x, y, z = sat_data['position']
                        clk = sat_data['clock']
                        sx, sy, sz, sclk = sat_data.get('sigma', (0,0,0,0))
                        
                        if not self.withDeviation or (sx==0 and sy==0 and sz==0 and sclk==0):
                            file.write(f"P{sat_id:>3} {x:13.6f} {y:13.6f} {z:13.6f} {clk:13.6f}\n")
                        else:
                            file.write(f"P{sat_id:>3} {x:13.6f} {y:13.6f} {z:13.6f} {clk:13.6f} {int(sx):2d} {int(sy):2d} {int(sz):2d} {int(sclk):3d}\n")
                        
                        if 'velocity' in sat_data and self.withVelocity:
                            vx, vy, vz = sat_data['velocity']
                            vclk = sat_data['vclk']
                            svx, svy, svz, svclk = sat_data.get('v_sigma', (0,0,0,0))
                            
                            if not self.withDeviation or (svx==0 and svy==0 and svz==0 and svclk==0):
                                file.write(f"V{sat_id:>3} {vx:13.6f} {vy:13.6f} {vz:13.6f} {vclk:13.6f}\n")
                            else:
                                file.write(f"V{sat_id:>3} {vx:13.6f} {vy:13.6f} {vz:13.6f} {vclk:13.6f} {int(svx):2d} {int(svy):2d} {int(svz):2d} {int(svclk):3d}\n")
            
            file.write("EOF\n")

    def write(self, output_path, new_data_df = None):
        self.write_header(output_path)
        if new_data_df is not None:
            # reconstruct self.data from new_data_df
            self.data = []
            for ts, group in new_data_df.groupby('timesystem'):
                epoch = {'time': ts, 'satellites': {}}
                for _, row in group.iterrows():
                    sat_id = row['sat_id']
                    x, y, z = row['x'], row['y'], row['z']
                    clk = row['clk']
                    sigma_x = row.get('sigma_x', 0)
                    sigma_y = row.get('sigma_y', 0)
                    sigma_z = row.get('sigma_z', 0)
                    sigma_clk = row.get('sigma_clk', 0)
                    
                    sat_data = {
                        'position': (x, y, z),
                        'clock': clk,
                        'sigma': (sigma_x, sigma_y, sigma_z, sigma_clk)
                    }
                    
                    if self.withVelocity:
                        vx, vy, vz = row.get('vx', 0), row.get('vy', 0), row.get('vz', 0)
                        vclk = row.get('vclk', 0)
                        svx = row.get('sigma_vx', 0)
                        svy = row.get('sigma_vy', 0)
                        svz = row.get('sigma_vz', 0)
                        svclk = row.get('sigma_vclk', 0)
                        
                        sat_data['velocity'] = (vx, vy, vz)
                        sat_data['vclk'] = vclk
                        sat_data['v_sigma'] = (svx, svy, svz, svclk)

                    epoch['satellites'][sat_id] = sat_data
                epoch['visible_sats'] = len(epoch['satellites'])
                self.data.append(epoch)
        self.write_data(output_path)

    def asDataFrame(self):
        """
        生成 DataFrame，并自动计算 J2000 历元秒 (gps_time)
        """
        data_list = []
        
        # 定义基准历元 (2000-01-01 12:00:00)
        j2000_epoch = TimeSystem.TimeSystem().from_ymdhms(2000, 1, 1, 12, 0, 0)

        for epoch in self.data:
            ts = epoch['time']
            # 直接通过两对象相减得到绝对秒数，完美兼容 KBR 数据时间戳
            gps_time = ts - j2000_epoch 

            for sat_id, sat_data in epoch['satellites'].items():
                x, y, z = sat_data.get('position', (0,0,0))
                clk = sat_data.get('clock', 0)
                
                row = {
                    'timesystem': ts,             
                    'gps_time': gps_time,         
                    'sat_id': sat_id,
                    'x': x, 'y': y, 'z': z, 'clk': clk
                }

                if self.withDeviation:
                    sx, sy, sz, sclk = sat_data.get('sigma', (0,0,0,0))
                    row.update({'sigma_x': sx, 'sigma_y': sy, 'sigma_z': sz, 'sigma_clk': sclk})
                
                if self.withVelocity:
                    vx, vy, vz = sat_data.get('velocity', (0,0,0))
                    vclk = sat_data.get('vclk', 0)
                    row.update({'vx': vx, 'vy': vy, 'vz': vz, 'vclk': vclk})
                    
                    if self.withDeviation:
                        svx, svy, svz, svclk = sat_data.get('v_sigma', (0,0,0,0))
                        row.update({'sigma_vx': svx, 'sigma_vy': svy, 'sigma_vz': svz, 'sigma_vclk': svclk})

                data_list.append(row)
                
        self.df = pd.DataFrame(data_list)
        return self.df

def read_sp3(file_path_list, start_time=None, end_time=None, returnsp3=False):
    """
    读取一个或多个 SP3 文件，并返回一个合并后的 DataFrame。
    参数:
        file_path_list: SP3 文件路径列表
        start_time: 可选的起始时间 (TimeSystem 对象)，用于过滤数据
        end_time: 可选的结束时间 (TimeSystem 对象)，用于过滤数据
        returnsp3: 可选的布尔值，用于指定是否返回 SP3 对象
    返回:
        包含所有 SP3 数据的 DataFrame，按时间排序
    """
    sp3_data_frames = []
    
    for file_path in file_path_list:
        sp3 = SP3(file_path)
        df = sp3.asDataFrame()
        
        if start_time:
            df = df[df['timesystem'] >= start_time]
        if end_time:
            df = df[df['timesystem'] <= end_time]
        
        sp3_data_frames.append(df)
    
    if sp3_data_frames:
        combined_df = pd.concat(sp3_data_frames).sort_values(by='timesystem').reset_index(drop=True)
        true_start_time = combined_df['timesystem'].min()
        sp3.header['time'] = true_start_time
        sp3.header['gps_week'] = true_start_time.gps_week
        sp3.header['gps_seconds'] = true_start_time.gps_sow
        sp3.header['mjd'] = true_start_time.mjd_int
        sp3.header['mjd_seconds'] = true_start_time.mjd_frac

        sp3.header['num_epochs'] = combined_df['timesystem'].nunique()

        print(sp3.header)
        print(sp3.header['num_epochs'], sp3.header['time'])
        sp3.data = []  # 清空原有数据，重新构建

        if returnsp3:
            return sp3, combined_df
        else:
            return combined_df
    else:
        if returnsp3:
            return None, pd.DataFrame()
        else:
            return pd.DataFrame()
        
