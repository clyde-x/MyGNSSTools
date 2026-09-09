import numpy as np 
import math 


import numpy as np

def window_interpolate(all_times, all_coords, target_time, max_half_window=4):
    """
    自适应窗口的拉格朗日插值封装。
    当边界数据不足时，自动缩小半窗口大小，最低退化为前后各1个点（线性插值）。
    """
    all_times = np.array(all_times)
    all_coords = np.array(all_coords)
    
    # 1. 寻找大于目标时间的第一个点的索引
    idx_right = np.searchsorted(all_times, target_time)
    
    # --- 提前处理正好命中的情况 ---
    if idx_right < len(all_times) and all_times[idx_right] == target_time:
        return all_coords[idx_right], 0.0
    if idx_right > 0 and all_times[idx_right - 1] == target_time:
        return all_coords[idx_right - 1], 0.0

    # 2. 检查是否发生外推 (Extrapolation)
    # GNSS 精密定位通常严格禁止外推，只允许内插
    if idx_right == 0:
        raise ValueError(f"目标时间 {target_time} 早于第一条记录，拒绝外推！")
    if idx_right == len(all_times):
        raise ValueError(f"目标时间 {target_time} 晚于最后一条记录，拒绝外推！")

    # 3. 动态计算可用的最大对称半窗口大小
    left_available = idx_right                     # 目标时间左侧有多少个点
    right_available = len(all_times) - idx_right   # 目标时间右侧有多少个点
    
    # 核心逻辑：取 [预设窗口, 左侧可用, 右侧可用] 三者中的最小值
    dynamic_half = min(max_half_window, left_available, right_available)
    
    # 如果 dynamic_half == 1，截取的就是前后各1个点，Neville 算法会自动执行线性插值
    if dynamic_half < 1:
        raise ValueError("严重错误：连前后各1个点都凑不齐，无法进行线性插值！")

    # 4. 按照动态计算的窗口截取数据
    start_idx = idx_right - dynamic_half
    end_idx = idx_right + dynamic_half
    
    local_X = all_times[start_idx : end_idx]
    local_Y = all_coords[start_idx : end_idx]
    
    # 5. 时间中心化 (防止绝对时间数值过大导致拉格朗日乘积精度丢失)
    t0 = local_X[0]
    local_X_centered = local_X - t0
    target_time_centered = target_time - t0
    
    # 6. 调用底层的 Neville 算法
    result, err = lagrange_interpolation_neville(local_X_centered, local_Y, target_time_centered)
    
    return result, err


# --- 附带：原封不动调用的底层 Neville 算法 ---
def lagrange_interpolation_neville(X, Y, x):
    n = len(X)
    k = n // 2
    if x == X[k]: return Y[k], 0.0
    if k > 0 and x == X[k-1]: return Y[k-1], 0.0
    if k > 0 and abs(x - X[k-1]) < abs(x - X[k]): k = k - 1
        
    Q, D = list(Y), list(Y)
    y = Y[k]
    k -= 1 
    err = 0.0
    
    for j in range(1, n):
        for i in range(n - j):
            if X[i] == X[i+j]: raise ValueError("存在重复时间戳！")
            del_val = (Q[i+1] - D[i]) / (X[i] - X[i+j])
            D[i] = (X[i+j] - x) * del_val
            Q[i] = (X[i] - x) * del_val
            
        if 2 * (k + 1) < n - j:
            err = Q[k+1]
        else:
            err = D[k]
            k -= 1
        y += err
        
    return y, err


from scipy.spatial.transform import Rotation, Slerp

def batch_quaternion_interpolate(times_array, quat_array, target_times):
    """
    利用 SciPy 进行批量的四元数序列插值
    
    参数:
    times_array: 已知观测点的时间戳序列 (1D array)
    quat_array: 已知观测点的四元数序列，形状为 (N, 4)，格式要求 [x, y, z, w]
    target_times: 需要求解的目标时间戳序列 (1D array)
    
    返回:
    interpolated_quats: 插值结果，形状为 (M, 4)
    """
    # 1. 构造 Rotation 对象
    rotations = Rotation.from_quat(quat_array)
    
    # 2. 创建 Slerp 插值器
    # 它会自动处理符号跳变和最短路径
    slerp_interpolator = Slerp(times_array, rotations)
    
    # 3. 对目标时间数组进行批量插值
    target_rotations = slerp_interpolator(target_times)
    
    # 4. 导出为四元数数组
    interpolated_quats = target_rotations.as_quat()
    
    return interpolated_quats

'''
    # --- 使用示例 ---
    # 假设有三个时刻的姿态
    t_known = [0.0, 10.0, 20.0]
    q_known = np.array([
        [0.0, 0.0, 0.0, 1.0], 
        [0.1, 0.0, 0.0, 0.995], 
        [0.2, 0.0, 0.0, 0.98]
    ])

    # 需要插值的时刻 (比如 5.0秒 和 15.0秒)
    t_target = [5.0, 15.0]

    q_result = batch_quaternion_interpolate(t_known, q_known, t_target)
    print(q_result)
'''

import numpy as np

def moving_avg_smooth(seq: np.ndarray, win=7) -> np.ndarray:
    n_sample, n_dim = seq.shape
    half_win = win // 2
    smooth_seq = np.zeros_like(seq)
    for i in range(n_sample):
        left = max(0, i - half_win)
        right = min(n_sample, i + half_win + 1)
        smooth_seq[i] = np.mean(seq[left:right], axis=0)
    return smooth_seq


def slow_motion_interpolate(t_target, t_arr: np.ndarray, nd_arr: np.ndarray, smooth_win=7):
    # 全局平滑
    smooth_nd = moving_avg_smooth(nd_arr, smooth_win)

    # 查找左右邻点
    idx = np.searchsorted(t_arr, t_target) - 1
    if idx < 0:
        return smooth_nd[0]
    if idx >= len(t_arr) - 1:
        return smooth_nd[-1]

    t0, t1 = t_arr[idx], t_arr[idx + 1]
    vec0 = smooth_nd[idx]
    vec1 = smooth_nd[idx + 1]

    alpha = (t_target - t0) / (t1 - t0)
    interp_vec = vec0 + alpha * (vec1 - vec0)
    return interp_vec
