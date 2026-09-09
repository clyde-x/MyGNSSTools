"""Dual-antenna RINEX observation fusion with valid-arc primary priority.

Version: 2026-07-25

Fusion policy
-------------
For every satellite independently, each antenna is first divided into tracking
arcs using its inferred sampling interval.  Only arcs whose inclusive duration
is at least ``min_continuous_time`` are valid.  At an epoch where both antennas
have a valid arc for the same satellite, antenna 1 (the primary antenna) is
selected.  Otherwise a valid antenna-2 observation fills the gap.

This preserves the public API of the legacy module.  The original implementation
is preserved verbatim at ``D:\\csu\\MyTools\\.history\\Merge_TwoAnts_Rinex_legacy_20260725.py``.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import CheckButtons

sys.path.insert(0, str(Path(r"D:\csu\MyTools\lib")))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import Rinex


DEFAULT_GAP_FACTOR = 1.5


def parse_time(line):
    """Parse a RINEX-3 epoch line (legacy public helper)."""
    year = int(line[2:6])
    month = int(line[7:9])
    day = int(line[10:12])
    hour = int(line[13:15])
    minute = int(line[16:18])
    second = float(line[19:29])
    return datetime(year, month, day, hour, minute, int(second), int((second % 1) * 1e6))


def _manual_read_rinex_obs(file_path, system='MIX'):
    """Legacy lightweight RINEX reader retained for compatibility."""
    with open(file_path, 'r', encoding='utf-8') as file:
        header, data, is_header, current_epoch = [], {}, True, None
        for line in file:
            if is_header:
                header.append(line)
                if 'END OF HEADER' in line:
                    is_header = False
                continue
            if line.startswith('>'):
                current_epoch = parse_time(line)
                data[current_epoch] = {'time': line[2:29].strip(), 'flag': line[31:32].strip(),
                                       'num_satellites': int(line[32:35].strip()), 'observations': {}}
            elif current_epoch is not None:
                sat_id = line[:3].strip()
                if system == 'GPS' and not sat_id.startswith('G'):
                    continue
                if system == 'BDS' and not sat_id.startswith('C'):
                    continue
                data[current_epoch]['observations'][sat_id] = line[:67]
        return header, data


def _sampling_interval(data, default=10.0):
    epochs = sorted(data)
    gaps = [(later - earlier).total_seconds() for earlier, later in zip(epochs, epochs[1:])
            if (later - earlier).total_seconds() > 0]
    return float(median(gaps)) if gaps else float(default)


def get_satellite_segments(data):
    """Return correctly reconstructed continuous segments for every satellite.

    The returned structure and keys match the legacy function.  ``duration`` is
    the elapsed time from first to last epoch, preserving the old field meaning.
    """
    interval = _sampling_interval(data)
    times_by_satellite = defaultdict(list)
    for epoch, info in data.items():
        for sat_id in info.get('observations', {}):
            times_by_satellite[sat_id].append(epoch)

    segments = defaultdict(list)
    for sat_id, times in times_by_satellite.items():
        times = sorted(set(times))
        start = previous = times[0]
        for current in times[1:]:
            if (current - previous).total_seconds() > DEFAULT_GAP_FACTOR * interval:
                segments[sat_id].append({'start_time': start, 'end_time': previous,
                                         'duration': (previous - start).total_seconds()})
                start = current
            previous = current
        segments[sat_id].append({'start_time': start, 'end_time': previous,
                                 'duration': (previous - start).total_seconds()})
    return segments


def _valid_epoch_membership(data, min_continuous_time):
    """Map each satellite to epochs that belong to a valid, inclusive arc."""
    interval = _sampling_interval(data)
    membership = defaultdict(set)
    all_times = defaultdict(list)
    for epoch, info in data.items():
        for sat_id in info.get('observations', {}):
            all_times[sat_id].append(epoch)

    for sat_id, times in all_times.items():
        times = sorted(set(times))
        begin = 0
        for index in range(1, len(times) + 1):
            is_break = index == len(times) or (times[index] - times[index - 1]).total_seconds() > DEFAULT_GAP_FACTOR * interval
            if not is_break:
                continue
            # A one-epoch observation represents one nominal sampling interval.
            duration = (times[index - 1] - times[begin]).total_seconds() + interval
            if duration >= min_continuous_time:
                membership[sat_id].update(times[begin:index])
            begin = index
    return membership


def merge_rinex_data_with_strategy1(data1, data2, min_continuous_time=10):
    """Fuse observations using valid-arc screening and satellite-wise priority.

    Args:
        data1: Primary-antenna observations in the legacy merge container.
        data2: Secondary-antenna observations in the legacy merge container.
        min_continuous_time: Minimum inclusive valid-arc duration in seconds.

    Returns:
        ``(merged_data, selection_records)`` with the same structure used by the
        former implementation.  At every epoch the selected observations equal
        the union of the two antennas' *valid-arc-filtered* satellite sets.
    """
    primary_valid = _valid_epoch_membership(data1, min_continuous_time)
    secondary_valid = _valid_epoch_membership(data2, min_continuous_time)
    merged_data = defaultdict(lambda: {'time': None, 'flag': None, 'num_satellites': 0, 'observations': {}})
    selection_records = defaultdict(dict)

    for epoch in sorted(set(data1).union(data2)):
        primary = data1.get(epoch, {}).get('observations', {})
        secondary = data2.get(epoch, {}).get('observations', {})
        info = merged_data[epoch]
        info['time'] = data1.get(epoch, {}).get('time') or data2.get(epoch, {}).get('time') or epoch
        info['flag'] = data1.get(epoch, {}).get('flag') or data2.get(epoch, {}).get('flag') or '0'

        for sat_id in set(primary).union(secondary):
            if sat_id in primary and epoch in primary_valid.get(sat_id, set()):
                info['observations'][sat_id] = primary[sat_id]
                selection_records[sat_id][epoch] = {'data_source': 'data1'}
            elif sat_id in secondary and epoch in secondary_valid.get(sat_id, set()):
                info['observations'][sat_id] = secondary[sat_id]
                selection_records[sat_id][epoch] = {'data_source': 'data2'}
        info['num_satellites'] = len(info['observations'])
    return merged_data, selection_records


def plot_arc_segments(data1_segments, data2_segments, merged_records):
    """Legacy interactive segment viewer retained for callers that use it."""
    fig, ax = plt.subplots(figsize=(15, 10))
    plt.subplots_adjust(left=.25)
    satellites = sorted(set(data1_segments) | set(data2_segments) | set(merged_records))
    colors = {'Data1': 'blue', 'Data2': 'green', 'Merged': 'red'}
    for index, sat_id in enumerate(satellites):
        base = index * 3
        for offset, label in ((2, 'Data1'), (1, 'Data2'), (0, 'Merged')):
            ax.add_patch(Rectangle((0, base + offset), 1, .8, color=colors[label], alpha=.3, label=label))
    ax.set_yticks([index * 3 + offset + .4 for index in range(len(satellites)) for offset in (2, 1, 0)])
    ax.set_yticklabels([f'{sat} {label}' for sat in satellites for label in ('Data1', 'Data2', 'Merged')])
    ax.set_xlabel('Time')
    check = CheckButtons(plt.axes([.05, .4, .1, .15]), ['Data1', 'Data2', 'Merged'], [True, True, True])
    def update_visibility(label):
        for patch in ax.patches:
            if patch.get_label() == label:
                patch.set_visible(not patch.get_visible())
        plt.draw()
    check.on_clicked(update_visibility)
    plt.show()


def filter_segments(segments, min_duration):
    """Return only legacy segment records meeting ``min_duration`` seconds."""
    return defaultdict(list, {sat_id: [segment for segment in records if segment['duration'] >= min_duration]
                              for sat_id, records in segments.items()})


def find_next_empty_segment(segments, current_epoch):
    """Legacy helper retained unchanged in purpose and return shape."""
    for index in range(len(segments) - 1):
        if segments[index]['end_time'] < current_epoch < segments[index + 1]['start_time']:
            return {'start_time': segments[index]['end_time'], 'end_time': segments[index + 1]['start_time']}
    return None


def find_empty_segment(segments, epoch):
    """Legacy helper retained for compatibility."""
    for segment in segments:
        if segment['end_time'] < epoch:
            return segment
    return None


def identify_empty_segments(segments, all_epochs):
    """Identify epochs not covered by any supplied segment dictionary."""
    empty, start, previous = [], None, None
    for epoch in all_epochs:
        covered = any(segment['start_time'] <= epoch <= segment['end_time']
                      for records in segments.values() for segment in records)
        if covered:
            if start is not None:
                empty.append({'start_time': start, 'end_time': previous})
                start = None
        elif start is None:
            start = epoch
        previous = epoch
    if start is not None:
        empty.append({'start_time': start, 'end_time': previous})
    return empty


def _manual_write_merged_rinex(file_path, header, data):
    """Legacy manual writer retained for compatibility."""
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(header)
        for epoch, info in sorted(data.items()):
            time = info['time'].strftime('%Y %m %d %H %M %S.%f') if isinstance(info['time'], datetime) else str(info['time'])
            file.write(f"> {time}  {info['flag']}{str(info['num_satellites']).rjust(3)}\n")
            for sat_id in sorted(info['observations']):
                file.write(str(info['observations'][sat_id]) + '\n')


def read_rinex_obs(file_path, system='MIX'):
    """Read observations through ``Rinex.py`` using the legacy merge container."""
    header = Rinex.readObsHead(file_path)
    native_data = Rinex.readObs(file_path, header)
    if header is None or native_data is None:
        raise ValueError(f'Unable to read RINEX observation file: {file_path}')
    prefixes = {'G'} if system == 'GPS' else {'C'} if system == 'BDS' else None
    if system not in ('GPS', 'BDS', 'MIX'):
        raise ValueError(f'Unsupported GNSS system selection: {system}')
    data = {}
    for epoch, satellites in native_data.items():
        observations = {sat_id: values for sat_id, values in satellites.items()
                        if prefixes is None or sat_id[:1] in prefixes}
        data[epoch] = {'time': epoch, 'flag': '0', 'num_satellites': len(observations), 'observations': observations}
    return header, data


def write_merged_rinex(file_path, header, data):
    """Write merged observations through ``Rinex.py`` (legacy API)."""
    native_data = {epoch: dict(sorted(info['observations'].items())) for epoch, info in sorted(data.items())
                   if info['observations']}
    if not native_data:
        raise ValueError('Merged RINEX contains no observation epochs')
    Rinex.writeObs(header, native_data, file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Fuse dual-antenna RINEX with antenna-2 priority on valid carrier arcs."
    )
    parser.add_argument("--ant2", type=Path, required=True, help="Primary (antenna 2) PCO-corrected RINEX.")
    parser.add_argument("--ant1", type=Path, required=True, help="Secondary (antenna 1) PCO-corrected RINEX.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-arc-seconds", type=float, default=50.0)
    args = parser.parse_args()

    header2, data2 = read_rinex_obs(str(args.ant2), system='MIX')
    _, data1 = read_rinex_obs(str(args.ant1), system='MIX')
    merged_data, records = merge_rinex_data_with_strategy1(
        data2, data1, min_continuous_time=args.minimum_arc_seconds
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_merged_rinex(str(args.output), header2, merged_data)
    print(
        f"policy=priority epochs={len(merged_data)} output={args.output} "
        f"selection_records={len(records)}"
    )
