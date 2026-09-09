"""
高级高精度时间传递（CPTT）链路评估工具 - 综合完整版（严格数据模式）
引入 PSD（功率谱密度）和动态高通滤波，并在稳定性评估中同时保留 TDEV (时间稳定性) 和 ADEV (频率稳定性)。
采用 3x2 的 6 宫格画幅全面展示各项性能。
注意：本程序严格依赖真实数据，若文件不存在将直接报错退出。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import allantools
from scipy import signal
import os
import sys

# ================================================================
# 1. 加载真实观测数据 (严格模式：无文件即退出)
# ================================================================
outdir = './out/group2/'
file_path = os.path.join(outdir, 'clock_kf_if.csv')

# 严格数据检查
if not os.path.exists(file_path):
    print(f"\n❌ 严重错误: 未找到真实观测数据文件 '{file_path}'。")
    print("为保证评估结论的严谨性，本程序拒绝生成或使用模拟数据。")
    print("请检查数据路径是否正确。程序已强制中止。\n")
    sys.exit(1)

print(f"✅ 成功找到真实数据文件: '{file_path}'，正在进行链路性能评估...")

# 加载数据
df = pd.read_csv(file_path)
df['time'] = pd.to_datetime(df['time'])

c = 299792458.0  # 光速 [m/s]
clock_m = df['clock_m'].values
clock_s = clock_m / c

dt = 30.0  # 您的数据采样率（秒）
rate = 1.0 / dt
N = len(clock_s)
time_axis = np.arange(N) * dt

# ================================================================
# 2. 功率谱密度 (PSD) 分析
# ================================================================
f, Pxx = signal.welch(clock_s, fs=rate, window='hann', nperseg=N//8, scaling='density')

# ================================================================
# 3. 动态高通滤波 (剥离晶振物理游走)
# ================================================================
cutoff_freq = 1.0 / 1000.0 
nyquist = 0.5 * rate
normal_cutoff = cutoff_freq / nyquist
b, a = signal.butter(2, normal_cutoff, btype='highpass', analog=False)

filtered_residuals_s = signal.filtfilt(b, a, clock_s)
rms_residuals_ps = np.sqrt(np.mean(filtered_residuals_s**2)) * 1e12

# ================================================================
# 4. 历元间平滑度 (Epoch-to-epoch Jitter)
# ================================================================
epoch_diff_s = np.diff(clock_s)
std_epoch_diff_ps = np.std(epoch_diff_s) * 1e12

# ================================================================
# 5. 计算 TDEV 和 ADEV
# ================================================================
# TDEV: 时间偏差 (Time Deviation) - 评估时间同步稳定性
taus_t, tdevs, _, _ = allantools.tdev(clock_s, rate=rate, data_type='phase', taus='decade')
# OADEV: 重叠Allan偏差 (Overlapping Allan Deviation) - 评估频率稳定性
taus_a, adevs, _, _ = allantools.oadev(clock_s, rate=rate, data_type='phase', taus='decade')

# ================================================================
# 6. 高级数据可视化 (3x2 六宫格布局)
# ================================================================
fig = plt.figure(figsize=(16, 14))
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# (1) 原始相对钟差 (时域)
ax1 = plt.subplot(3, 2, 1)
ax1.plot(time_axis / 3600, clock_s * 1e9, 'b-', linewidth=1)
ax1.set_title('Raw Relative Clock (Time Domain)', fontweight='bold')
ax1.set_xlabel('Time (Hours)')
ax1.set_ylabel('Time Offset (ns)')
mean_val = np.mean(clock_s * 1e9)
std_val = np.std(clock_s * 1e9)
ax1.text(0.05, 0.85, f'Mean: {mean_val:.2f} ns\nStd: {std_val:.2f} ns', transform=ax1.transAxes, color='darkred', bbox=dict(facecolor='white', alpha=0.8))

# (2) 高通滤波残差 (算法真实精度)
ax2 = plt.subplot(3, 2, 2)
ax2.plot(time_axis / 3600, filtered_residuals_s * 1e12, 'g-', linewidth=1, alpha=0.8)
ax2.axhline(0, color='black', linewidth=0.8)
ax2.axhline(rms_residuals_ps, color='red', linestyle='--', label=f'+1 RMS ({rms_residuals_ps:.1f} ps)')
ax2.axhline(-rms_residuals_ps, color='red', linestyle='--', label=f'-1 RMS ({-rms_residuals_ps:.1f} ps)')
ax2.set_title(f'High-pass Residuals (> {1/cutoff_freq:.0f}s removed)', fontweight='bold')
ax2.set_xlabel('Time (Hours)')
ax2.set_ylabel('Residuals (ps)')
ax2.legend()

# (3) 功率谱密度 PSD (频域)
ax3 = plt.subplot(3, 2, 3)
ax3.loglog(f, Pxx, 'purple', linewidth=1.5)
ax3.set_title('Power Spectral Density (Frequency Domain)', fontweight='bold')
ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel('PSD ($s^2$/Hz)')
ax3.axvline(x=cutoff_freq, color='gray', linestyle='--', label='Filter Cutoff')
ax3.legend()

# (4) 历元间跳变 Jitter (时域导数)
ax4 = plt.subplot(3, 2, 4)
ax4.plot(time_axis[1:] / 3600, epoch_diff_s * 1e12, 'c-', linewidth=0.8, alpha=0.7)
ax4.axhline(0, color='black', linewidth=0.8)
ax4.axhline(std_epoch_diff_ps, color='red', linestyle='--', label=f'+1 STD ({std_epoch_diff_ps:.1f} ps)')
ax4.axhline(-std_epoch_diff_ps, color='red', linestyle='--', label=f'-1 STD ({-std_epoch_diff_ps:.1f} ps)')
ax4.set_title('Epoch-to-epoch Jitter (Phase Rate)', fontweight='bold')
ax4.set_xlabel('Time (Hours)')
ax4.set_ylabel('Jitter (ps/epoch)')
ax4.legend()

# (5) TDEV (时间稳定性)
ax5 = plt.subplot(3, 2, 5)
ax5.loglog(taus_t, tdevs, 'b-o', markersize=4, linewidth=1.5)
ref_line_tdev = [tdevs[0] * np.sqrt(t / taus_t[0]) for t in taus_t]
ax5.loglog(taus_t, ref_line_tdev, 'k--', alpha=0.5, label=r'$\propto \tau^{1/2}$')
ax5.set_title('Time Deviation (TDEV)', fontweight='bold')
ax5.set_xlabel('Averaging Time $\tau$ (s)')
ax5.set_ylabel('TDEV (s)')
ax5.legend()

# (6) ADEV (频率稳定性)
ax6 = plt.subplot(3, 2, 6)
ax6.loglog(taus_a, adevs, 'r-o', markersize=4, linewidth=1.5)
ref_line_adev = [adevs[0] * np.sqrt(taus_a[0] / t) for t in taus_a]
ax6.loglog(taus_a, ref_line_adev, 'k--', alpha=0.5, label=r'White FM $\propto \tau^{-1/2}$')
ax6.set_title('Overlapping Allan Deviation (OADEV)', fontweight='bold')
ax6.set_xlabel('Averaging Time $\tau$ (s)')
ax6.set_ylabel('ADEV')
ax6.legend()

plt.tight_layout()
plt.savefig(outdir + 'advanced_cptt_eval_full.png', dpi=150)
plt.show()

# ================================================================
# 7. 输出核心评价指标
# ================================================================
print("=" * 60)
print(" 🚀 高级时间传递链路性能评估报告 (包含 TDEV & ADEV) 🚀")
print("=" * 60)
print(f"[1] 高通滤波残差 RMS (Link Precision)   : {rms_residuals_ps:.2f} ps (皮秒)")
print("-" * 60)
print(f"[2] 历元间钟差跳变 STD (Epoch Jitter) : {std_epoch_diff_ps:.2f} ps (皮秒)")
print("-" * 60)
print(f"[3] TDEV @ {taus_t[0]:.0f}s (短期时间稳定度)         : {tdevs[0]*1e12:.2f} ps")
print("-" * 60)
print(f"[4] ADEV @ {taus_a[0]:.0f}s (短期频率稳定度)         : {adevs[0]:.2e}")
print("=" * 60)