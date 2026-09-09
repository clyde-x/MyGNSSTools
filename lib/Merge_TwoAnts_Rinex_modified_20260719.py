"""
-*- coding: utf-8 -*-
@Time  : 2024/08/21 15:11
# File       : Merge_TwoAnts_Rinex.py
# Time       ：2024/8/19 15:11
# Author     ：TianShun
# Description：
"""

import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
from datetime import timedelta
from matplotlib.patches import Rectangle
from matplotlib.widgets import CheckButtons
# import PlotAnalysis as pa
# import logging

def parse_time(line):
    year = int(line[2:6])
    month = int(line[7:9])
    day = int(line[10:12])
    hour = int(line[13:15])
    minute = int(line[16:18])
    second = float(line[19:29])
    return datetime(year, month, day, hour, minute, int(second), int((second % 1) * 1e6))


def read_rinex_obs(file_path, system='MIX'):
    with open(file_path, 'r', encoding='utf-8') as file:
        header = []
        data = {}
        is_header = True
        epoch = None
        obs_types = {}

        for line in file:
            if is_header:
                header.append(line)
                if "SYS / # / OBS TYPES" in line:
                    sys_id = line[0]
                    if sys_id in "GRECJ":
                        obs_types.setdefault(sys_id, []).extend(line[7:60].split())
                if "END OF HEADER" in line:
                    is_header = False
                continue

            # line = line.rstrip()  # 移除右侧的空白字符，避免重复调用 strip()
            if line.startswith('>'):

                time = parse_time(line)
                current_epoch = time

                time0 = line[2:29].strip()
                flag = line[31:32].strip()
                num_satellites = int(line[32:35].strip())
                data[current_epoch] = {
                    'time': time0,
                    'time_obj': time,
                    'epoch_line': line.rstrip('\n'),
                    'flag': flag,
                    'num_satellites': num_satellites,
                    'observations': {}
                }
            elif current_epoch:
                sat_id = line[0:3].strip()  # 去掉卫星ID周围的空格
                if system == 'GPS' and not sat_id.startswith('G'):
                    continue  # 跳过非GPS卫星
                if system == 'BDS' and not sat_id.startswith('C'):
                    continue  # 跳过非BDS卫星
                # Preserve the original fixed-width observation record. The
                # header is kept in the same order, so no reformatting is
                # needed and CSUPODS receives the exact source columns.
                data[current_epoch]['observations'][sat_id] = line.rstrip('\n')

        return header, data


# def get_satellite_segments(data):
#     segments = defaultdict(list)
#     last_epoch = None
#     current_segment = {}

#     for epoch in sorted(data.keys()):
#         for sat_id in data[epoch]['observations']:
#             if sat_id not in segments or last_epoch is None or (epoch - last_epoch).total_seconds() > 10:
#                 # 新弧段
#                 if sat_id in current_segment:
#                     segments[sat_id].append(current_segment[sat_id])
#                 current_segment[sat_id] = {
#                     'start_time': epoch,
#                     'end_time': epoch,
#                     'duration': 0
#                 }
#             else:
#                 # 延续当前弧段
#                 current_segment[sat_id]['end_time'] = epoch
#                 current_segment[sat_id]['duration'] += (epoch - last_epoch).total_seconds()
#         last_epoch = epoch

#     # 添加最后的弧段
#     for sat_id in current_segment:
#         segments[sat_id].append(current_segment[sat_id])

#     return segments

def get_satellite_segments(data, gap_threshold=25):
    segments = defaultdict(list)
    current_segments = {}

    for epoch in sorted(data.keys()):

        for sat_id in data[epoch]['observations']:

            seg = current_segments.get(sat_id)

            # 不存在当前弧段
            if seg is None:
                current_segments[sat_id] = {
                    'start_time': epoch,
                    'end_time': epoch,
                    'duration': 0
                }
                continue

            # 与当前弧段末尾时间差
            gap = (epoch - seg['end_time']).total_seconds()

            # 超过阈值 -> 新弧段
            if gap > gap_threshold:
                segments[sat_id].append(seg)

                current_segments[sat_id] = {
                    'start_time': epoch,
                    'end_time': epoch,
                    'duration': 0
                }

            # 延续当前弧段
            else:
                seg['duration'] += gap
                seg['end_time'] = epoch

    # flush 最后一个弧段
    for sat_id, seg in current_segments.items():
        segments[sat_id].append(seg)

    return segments


def merge_rinex_data_with_strategy1(data1, data2, min_continuous_time=10):
    merged_data = defaultdict(lambda: {
        'time': None,
        'flag': None,
        'num_satellites': 0,
        'observations': {}
    })

    # 获取卫星弧段，不对data1进行过滤
    segments1 = get_satellite_segments(data1)
    segments2 = get_satellite_segments(data2)

    all_epochs = sorted(set(data1.keys()).union(data2.keys()))
    selection_records = defaultdict(dict)

    # 预处理每个卫星在data1中的空白段
    satellite_empty_segments = defaultdict(list)
    for sat_id in set(segments1.keys()).union(segments2.keys()):
        # 获取data1的弧段并排序
        segs1 = sorted(segments1.get(sat_id, []), key=lambda x: x['start_time'])
        empty_segs = []

        # 生成空白段
        prev_end = None
        for seg in segs1:
            if prev_end is None:
                if seg['start_time'] > all_epochs[0]:
                    empty_segs.append({'start': all_epochs[0], 'end': seg['start_time'] - timedelta(seconds=1)})
            else:
                if seg['start_time'] > prev_end + timedelta(seconds=1):
                    empty_segs.append(
                        {'start': prev_end + timedelta(seconds=1), 'end': seg['start_time'] - timedelta(seconds=1)})
            prev_end = seg['end_time']
        if prev_end and prev_end < all_epochs[-1]:
            empty_segs.append({'start': prev_end + timedelta(seconds=1), 'end': all_epochs[-1]})
        satellite_empty_segments[sat_id] = empty_segs

    for epoch in all_epochs:
        # 合并基础信息
        merged_data[epoch]['time'] = data1.get(epoch, {}).get('time') or data2.get(epoch, {}).get('time')
        merged_data[epoch]['epoch_line'] = data1.get(epoch, {}).get('epoch_line') or data2.get(epoch, {}).get('epoch_line')
        merged_data[epoch]['flag'] = data1.get(epoch, {}).get('flag') or data2.get(epoch, {}).get('flag')

        satellites = set(data1.get(epoch, {}).get('observations', {})).union(
            data2.get(epoch, {}).get('observations', {}))

        for sat_id in satellites:
            # 优先使用data1的数据
            if sat_id in data1.get(epoch, {}).get('observations', {}):
                merged_data[epoch]['observations'][sat_id] = data1[epoch]['observations'][sat_id]
                selection_records[sat_id][epoch] = {'data_source': 'data1'}
            else:
                # 检查data2的弧段是否在空白段并满足连续时间
                in_valid_segment = False
                for seg in segments2.get(sat_id, []):
                    if seg['start_time'] <= epoch <= seg['end_time']:
                        # 检查是否在data1的空白段内
                        for empty in satellite_empty_segments[sat_id]:
                            if empty['start'] <= epoch <= empty['end']:
                                overlap_start = max(seg['start_time'], empty['start'])
                                overlap_end = min(seg['end_time'], empty['end'])
                                duration = (overlap_end - overlap_start).total_seconds()
                                if duration >= min_continuous_time:
                                    merged_data[epoch]['observations'][sat_id] = data2[epoch]['observations'][sat_id]
                                    selection_records[sat_id][epoch] = {'data_source': 'data2'}
                                    in_valid_segment = True
                                    break
                        if in_valid_segment:
                            break

        merged_data[epoch]['num_satellites'] = len(merged_data[epoch]['observations'])

    return merged_data, selection_records


def plot_arc_segments(data1_segments, data2_segments, merged_records):
    fig, ax = plt.subplots(figsize=(15, 10))
    plt.subplots_adjust(left=0.25)

    # 组织卫星数据
    sats = sorted(set(data1_segments.keys()) | set(data2_segments.keys()) | set(merged_records.keys()))
    y_ticks = []
    y_labels = []

    # 为每个卫星创建三个轨道
    for i, sat in enumerate(sats):
        base_y = i * 3
        # Data1原始弧段
        ax.add_patch(Rectangle((0, base_y + 2), 1, 0.8, color='blue', alpha=0.3, label='Data1'))
        # Data2原始弧段
        ax.add_patch(Rectangle((0, base_y + 1), 1, 0.8, color='green', alpha=0.3, label='Data2'))
        # 合并后的弧段
        ax.add_patch(Rectangle((0, base_y), 1, 0.8, color='red', alpha=0.3, label='Merged'))
        y_ticks.extend([base_y + 2.4, base_y + 1.4, base_y + 0.4])
        y_labels.extend([f'{sat} Data1', f'{sat} Data2', f'{sat} Merged'])

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Time')

    # 添加交互复选框
    check = CheckButtons(plt.axes([0.05, 0.4, 0.1, 0.15]),
                         ['Data1', 'Data2', 'Merged'],
                         [True, True, True])

    # 可见性控制逻辑
    def update_visibility(label):
        for art in ax.artists:
            if art.get_label() == label:
                art.set_visible(not art.get_visible())
        plt.draw()

    check.on_clicked(update_visibility)
    plt.show()


def filter_segments(segments, min_duration):
    filtered_segments = defaultdict(list)
    for sat_id, seg_list in segments.items():
        filtered_segments[sat_id] = [
            seg for seg in seg_list if seg['duration'] >= min_duration
        ]
    return filtered_segments

def find_next_empty_segment(segments, current_epoch):
    # 查找在当前历元之后的下一个空余弧段
    if not segments:
        return None

    for i, seg in enumerate(segments):
        if i < len(segments) - 1 and segments[i]['end_time'] < current_epoch < segments[i + 1]['start_time']:
            return {
                'start_time': segments[i]['end_time'],
                'end_time': segments[i + 1]['start_time']
            }
    return None


def find_empty_segment(segments, epoch):
    for seg in segments:
        if seg['end_time'] < epoch:
            return seg
    return None


def identify_empty_segments(segments, all_epochs):
    empty_segments = []
    current_empty_start = None
    last_end_time = None

    for epoch in all_epochs:
        if any(seg['start_time'] <= epoch <= seg['end_time'] for seg in segments.values()):
            if current_empty_start:
                empty_segments.append({'start_time': current_empty_start, 'end_time': last_end_time})
                current_empty_start = None
        else:
            if not current_empty_start:
                current_empty_start = epoch
        last_end_time = epoch

    if current_empty_start:
        empty_segments.append({'start_time': current_empty_start, 'end_time': last_end_time})

    return empty_segments


# 将合并后的数据写回一个新的RINEX文件
def write_merged_rinex(file_path, header, data):
    obs_types = {}
    for line in header:
        if "SYS / # / OBS TYPES" in line and line[0] in "GRECJ":
            obs_types.setdefault(line[0], []).extend(line[7:60].split())
    with open(file_path, 'w', encoding='utf-8') as file:
        for line in header:
            file.write(line)

        for epoch, info in sorted(data.items()):
            # Preserve the original fixed-column epoch record and only update
            # the satellite count (columns 33-35). This avoids subtle parser
            # differences in CSUPODS caused by reconstructing the timestamp.
            dt = info.get('time_obj', epoch)
            second = dt.second + dt.microsecond / 1e6
            updated_epoch_line = (
                f"> {dt.year:4d} {dt.month:02d} {dt.day:02d} {dt.hour:02d} "
                f"{dt.minute:02d}{second:11.7f}  {info['flag']} "
                f"{info['num_satellites']:02d}       0.000000000000"
            )
            updated_epoch_line = updated_epoch_line.ljust(80) + "\n"
            file.write(updated_epoch_line)
            for sat_id in sorted(info['observations']):
                record = info['observations'][sat_id]
                file.write(record + ('\n' if not record.endswith('\n') else ''))

# 示例使用：
if __name__ == '__main__':
    # file_path1 = "D:\Study_Project\GNSS_BDS\CSUAPPS\CSUPODSApp\BIN\inputdata\Obs\DRO\-X-Ant2_DRO_L_GNSS_20241226_H_N_305_10.rnx"
    # file_path2 = "D:\Study_Project\GNSS_BDS\CSUAPPS\CSUPODSApp\BIN\inputdata\Obs\DRO\+X-Ant1_DRO_L_GNSS_20241226_H_N_305_10.rnx"
    # file_path1 = "D:\Study_Project\GNSS_rinex3_pco\CSUAPPS\CSUPODSApp\BIN\TempData\DRO\+X-DRO_1_RecObs_Dyn_20241227.rnx"
    # file_path2 = "D:\Study_Project\GNSS_rinex3_pco\CSUAPPS\CSUPODSApp\BIN\TempData\DRO\-X-DRO_2_RecObs_Dyn_20241227.rnx"
    file_path2 = r"D:\csu\dataset\DROL_data\-X-Ant2_PCO_DRO_L_GNSS_20241226_H_N_305_10.rnx"
    file_path1 = r"D:\csu\dataset\DROL_data\+X-Ant1_PCO_DRO_L_GNSS_20241226_H_N_305_10.rnx"

    # merged_outfile = "D:\Study_Project\GNSS_fusion\CSUAPPS\CSUPODSApp\BIN\inputdata\Obs\DRO\ANT3\DRO_3_20241227_AfterPCO_MergeDeep1_20-20.rnx"
    merged_outfile = r"D:\csu\dataset\DROL_data\dro_pco_merged.rnx"

    header1, data1 = read_rinex_obs(file_path1, system='MIX')
    header2, data2 = read_rinex_obs(file_path2, system='MIX')

    headerout = header1.copy()
    flag = False
    for idx, headerline in enumerate(headerout):
        if "SYS / # / OBS TYPES" in headerline and headerline.startswith('G'):
            headerline = 'G    4 C1C L1C C2S L2S                                      SYS / # / OBS TYPES\n'
            headerout[idx] = headerline
        elif "SYS / # / OBS TYPES" in headerline and headerline.startswith('C'):
            headerline = 'C    4 C1C L1C C2S L2S                                      SYS / # / OBS TYPES\n'
            headerout[idx] = headerline
                
    # pa.plot_satellite_visibility_plotly1(segments1)
    # pa.plot_satellite_visibility(segments1)
    # merged_data, selection_records, merged_segments = merge_rinex_data_with_strategy4(data1, data2, min_duration=40, min_continuous_time=40)
    merged_data, records  = merge_rinex_data_with_strategy1(data1, data2)
    # pa.plot_selection_records(selection_records)
    write_merged_rinex(merged_outfile, headerout, merged_data)
    # Plot the segments
    # pa.plot_satellite_segments(segments1, segments2, merged_segments)
    # pa.plot_satellite_segments_plotly1(segments1, segments2, merged_segments)
    # pa.plot_selection_records(selection_records)
    # 获取弧段信息
    # seg1 = get_satellite_segments(data1)
    # seg2 = get_satellite_segments(data2)
    #
    # # 绘制可视化图表
    # plot_arc_segments(seg1, seg2, records)
