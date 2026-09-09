'''
parse_file(filename) return df, comments
'''

import numpy as np
import pandas as pd 
import sys
import os
sys.path.append("../lib")

import TimeSystem
import Rinex
import SatID 
from dataclasses import dataclass

def parse_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    comments = []
    line_count = 0
    for line in lines:
        if line.startswith('%'):
            comments.append(line)
            line_count += 1
        else:
            break

    colum_name = comments[-1]
    colum_name = colum_name.split()[1:]  # 去掉开头的 '%' 和 'TIME' 列
    colum_name[0] = "timesystem"  # 将第一列重命名为 timesystem
    print(colum_name)
    
    data_records = []
    for line in lines[line_count:]:
        time = line[:24]
        timesystem = TimeSystem.TimeSystem().from_str(time)
        line = line[24:]
        data = line.split()
        data = [float(x) for x in data]
        if len(data) != len(colum_name) -1:
            print(f"警告：数据行长度 {len(data)} 与列名长度 {len(colum_name)} 不匹配！")
            print(data)
            continue
        else:
            data.insert(0, timesystem)  # 将时间字符串插入到数据列表的开头
            data_records.append(data)
    
    df = pd.DataFrame(data_records, columns=colum_name)

    return df, comments
