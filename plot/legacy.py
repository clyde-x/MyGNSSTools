"""
2025-10-29    v 0.1
画各种图像的，目前有点繁杂
"""

from matplotlib import gridspec
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os 
import sys
from pathlib import Path

from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

# ``PlotTool`` used to live in ``lib``.  Keep dependency lookup independent
# of the process working directory after consolidation into ``plot``.
_LIB_DIR = Path(__file__).resolve().parents[1] / "lib"
if str(_LIB_DIR) not in sys.path:
  sys.path.insert(0, str(_LIB_DIR))
import TimeSystem

def convert_time_column(df, time_col='time'):
  for i in range(len(df)):
    time = df.loc[i,time_col]
    df.loc[i,time_col] = time.datetime
  return df

def plot_error(config):
  name1 = config['name1']
  name2 = config['name2']
  analysis_path = config['analysis_path']
  POD_type = config['POD_type']
  df_path = os.path.join(analysis_path, f'{POD_type}_diff.csv')
  df = pd.read_csv(df_path)
  if os.path.exists(os.path.join(analysis_path, 'imgs')) is False:
    os.makedirs(os.path.join(analysis_path, 'imgs'))
  img_path = os.path.join(analysis_path, 'imgs')

  if_v = ('vx_err' in df.columns)
  if_dt = (np.abs(df['clk_err_ms'].mean())<100)
  df['time'] = pd.to_datetime(df['time'])
  
  
  plt.figure(figsize=(10, 6))
  plt.plot(df['time'], df['x_err_mm'], label='Error X(mm)')
  plt.plot(df['time'], df['y_err_mm'], label='Error Y(mm)')
  plt.plot(df['time'], df['z_err_mm'], label='Error Z(mm)')
  plt.xlabel('Time')
  plt.ylabel('Error (mm)')
  plt.title(f'{name1} - {name2} Position Error')
  plt.legend()
  rms_x = np.sqrt(np.mean(df['x_err_mm']**2))
  rms_y = np.sqrt(np.mean(df['y_err_mm']**2))
  rms_z = np.sqrt(np.mean(df['z_err_mm']**2))

  # 添加多行文本
  stats_text = f'RMS Errors(mm):\nX: {rms_x:.4f}\nY: {rms_y:.4f}\nZ: {rms_z:.4f}\nOverall: {np.sqrt(rms_x**2 + rms_y**2 + rms_z**2):.4f}'
  plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
          fontsize=10, verticalalignment='top',
          bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
  print("3d position RMS error(mm): ", np.sqrt(rms_x**2 + rms_y**2 + rms_z**2))
  if config['savefig']:
    plt.savefig(os.path.join(img_path,f'{name1}_{name2}_position_error.png'))

  if if_v:  
    plt.figure(figsize=(10, 6))
    plt.plot(df['time'], df['vx_err'], label='Error Vx')
    plt.plot(df['time'], df['vy_err'], label='Error Vy')
    plt.plot(df['time'], df['vz_err'], label='Error Vz')
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.title(f'{name1} - {name2} Velocity Error')
    plt.legend()
    rms_vx = np.sqrt(np.mean(df['vx_err']**2))
    rms_vy = np.sqrt(np.mean(df['vy_err']**2))
    rms_vz = np.sqrt(np.mean(df['vz_err']**2))
    # 添加多行文本
    stats_text = f'RMS Errors(mm):\nVx: {rms_vx:.4f}\nVy: {rms_vy:.4f}\nVz: {rms_vz:.4f}\nOverall: {np.sqrt(rms_vx**2 + rms_vy**2 + rms_vz**2):.4f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    if config['savefig']:
      plt.savefig(os.path.join(img_path,f'{name1}_{name2}_velocity_error.png'))
  
  if if_dt:
    plt.figure(figsize=(10, 6))
    plt.plot(df['time'], df['clk_err_ms'], label='Clock Error (ms)')
    plt.xlabel('Time')
    plt.ylabel('Clock Error (ms)')
    plt.title(f'{name1} - {name2} Clock Error')
    plt.legend()
    rms_clk = np.sqrt(np.mean(df['clk_err_ms']**2))
    stats_text = f'RMS Clock Error(ms): {rms_clk:.4f} \nSTD Clock Error(ms): {df["clk_err_ms"].std():.4f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    print("clk std error(ms): ", df["clk_err_ms"].std())
    if config['savefig']:
      plt.savefig(os.path.join(img_path,f'{name1}_{name2}_clock_error.png'))

def visualize_spp_log(df, config):
    """
    可视化SPP处理结果
    :param df: DataFrame包含SPP处理结果
    """
    # 创建图表布局
    plt.figure(figsize=(16, 20))
    
    # 1. 数据合格率统计
    # plt.subplot(4, 2, 1)
    # mode_counts = df['mode'].value_counts()
    # plt.pie(mode_counts, labels=mode_counts.index, autopct='%1.1f%%', 
    #         colors=['#ff9999','#66b3ff'], startangle=90)
    
    # plt.axis('equal')
    
    # 2. 卫星数量时间序列
    plt.subplot(3, 2, 1)
    plt.plot(df['sat_num'], 'b-', linewidth=1)
    plt.title('Visible Sat Num')
    plt.xlabel('Epoch')
    plt.ylabel('Sat Num')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 3. 钟差补偿时间序列
    plt.subplot(3, 2, 2)
    plt.plot( df['clk_offset'], 'g-', linewidth=1)
    plt.title('Clk Offset')
    plt.xlabel('Time')
    plt.ylabel('Clk Offset')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 4. 位置增量时间序列
    # plt.subplot(4, 2, 4)
    # plt.plot(df['delta'], 'r-', linewidth=1)
    # plt.title('delta')
    # plt.xlabel('time')
    # plt.ylabel('delta')
    # plt.grid(True, linestyle='--', alpha=0.7)
    
    # 5. 迭代次数统计
    plt.subplot(3, 2, 3)
    sns.histplot(df['itr'], bins=range(1, int(df['itr'].max())+2), 
                 kde=False, color='purple')
    plt.title('itr')
    plt.xlabel('itr')
    plt.ylabel('frequency')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 6. 残差分布箱线图
    plt.subplot(3, 2, 4)
    # 展开残差列表
    all_prefit = [val for sublist in df['prefitC'] for val in sublist]
    plt.boxplot(all_prefit, vert=True, patch_artist=True)
    plt.title('prefitC')
    plt.ylabel('prefitC')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 7. 精度与PDOP关系
    plt.subplot(3, 2, 5)
    # 计算平均精度和PDOP
    avg_precise = [np.mean(lst) for lst in df['precise']]
    avg_pdop = [np.mean(lst) for lst in df['PDOD']]
    
    plt.scatter(avg_pdop, avg_precise, c=df['sat_num'], 
                cmap='viridis', alpha=0.7)
    plt.colorbar(label='Sat Num')
    plt.title('Precise vs PDOP')
    plt.xlabel('PDPD')
    plt.ylabel('Precise')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 8. 仰角与残差关系
    plt.subplot(3, 2, 6)
    # 展开仰角和残差
    all_ele = [val for sublist in df['ele'] for val in sublist]
    all_prefit = [val for sublist in df['prefitC'] for val in sublist]
    
    # 使用hexbin展示二维分布
    plt.hexbin(all_ele, all_prefit, gridsize=30, cmap='plasma', mincnt=1)
    plt.colorbar(label='Counts')
    plt.title('ele vs prefitC')
    plt.xlabel('ele(°)')
    plt.ylabel('prefitC(m)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 时间格式化
    # for i in [2, 3, 4]:
    #     plt.subplot(4, 2, i)
    #     plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%M'))
    #     plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    
    plt.tight_layout()
    # plt.show()
    if config['savefig']:
      plt.savefig(os.path.join(config['output_path'],'spp_log_visualization.png'), dpi=300)

# 示例调用
# visualize_spp_results(spp_df)

def visualize_spp_info(df, config):
    """
    可视化SPP关键指标时间序列
    :param df: DataFrame包含SPP处理结果
    """
    # 设置图表风格
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 12))

    df = convert_time_column(df, time_col='time')
    
    # 1. 卫星数量时间序列
    plt.subplot(3, 1, 1)
    plt.plot(df['time'], df['sat_num'], 'b-', linewidth=1.5, marker='o', markersize=4)
    plt.title('Sat Num', fontsize=14)
    plt.xlabel('Time')
    plt.ylabel('Sat Num')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, max(df['sat_num'])+2)
    
    # 2. PDOP时间序列
    plt.subplot(3, 1, 2)
    plt.plot(df['time'], df['PDOD'], 'r-', linewidth=1.5)
    plt.title('PDOP', fontsize=14)
    plt.xlabel('Time')
    plt.ylabel('PDOP')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加PDOP质量阈值线
    plt.axhline(y=3, color='g', linestyle='--', alpha=0.7)
    plt.axhline(y=6, color='y', linestyle='--', alpha=0.7)
    plt.axhline(y=8, color='r', linestyle='--', alpha=0.7)
    
    # 3. 定位精度时间序列
    plt.subplot(3, 1, 3)
    plt.plot(df['time'], df['precise'], 'g-', linewidth=1.5)
    plt.title('Precise', fontsize=14)
    plt.xlabel('Time')
    plt.ylabel('Precise')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加精度阈值线
    plt.axhline(y=5, color='g', linestyle='--', alpha=0.7)
    plt.axhline(y=10, color='y', linestyle='--', alpha=0.7)
    plt.axhline(y=20, color='r', linestyle='--', alpha=0.7)

    
    # 时间格式化
    for i in range(1, 4):
        plt.subplot(3, 1, i)
        plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # # 4. 创建关系分析图
    # plt.figure(figsize=(15, 5))
    
    # # 卫星数量与PDOP关系
    # plt.subplot(1, 3, 1)
    # sns.regplot(x='sat_num', y='PDOD', data=df, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
    # plt.title('SatNum vs PDOP')
    # plt.xlabel('SatNum')
    # plt.ylabel('PDOP')
    
    # # 卫星数量与定位精度关系
    # plt.subplot(1, 3, 2)
    # sns.regplot(x='sat_num', y='precise', data=df, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
    # plt.title('SatNum vs Precise')
    # plt.xlabel('SatNum')
    # plt.ylabel('Precise')
    
    # # PDOP与定位精度关系
    # plt.subplot(1, 3, 3)
    # sns.regplot(x='PDOD', y='precise', data=df, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
    # plt.title('PDOP vs Precise')
    # plt.xlabel('PDOP')
    # plt.ylabel('Precise')
    
    # plt.tight_layout()
    #plt.show()
    if config['savefig']:
      plt.savefig(os.path.join(config['output_path'],'spp_info_visualization.png'), dpi=300)

# 示例调用
# visualize_spp_summary(spp_df)

import Sp3Tools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.colors import Normalize, LinearSegmentedColormap
def VisualizeOrbit3d(df, earth_radius=6371):
    df = convert_time_column(df, time_col='time')
    # 1. 准备数据
    unique_sats = df['sat_id'].unique()
    sat_count = len(unique_sats)
    
    # 创建颜色映射
    colors = plt.cm.jet(np.linspace(0, 1, sat_count))
    sat_colors = dict(zip(unique_sats, colors))
    
    # 2. 创建3D图表
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 3. 绘制地球
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    
    # 地球表面
    x_earth = earth_radius * np.outer(np.cos(u), np.sin(v))
    y_earth = earth_radius * np.outer(np.sin(u), np.sin(v))
    z_earth = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x_earth, y_earth, z_earth, 
                    color='lightblue', alpha=0.3, 
                    edgecolor='none', rstride=5, cstride=5)
    
    # 添加赤道线
    equator = Circle((0, 0), earth_radius, color='blue', alpha=0.2)
    ax.add_patch(equator)
    art3d.pathpatch_2d_to_3d(equator, z=0, zdir="z")
    
    # 4. 绘制卫星轨道
    # 存储所有卫星的轨迹数据
    orbits = {}
    
    for sat in unique_sats:
        sat_df = df[df['sat_id'] == sat]
        x = sat_df['x'].values
        y = sat_df['y'].values
        z = sat_df['z'].values
        
        # 保存轨道数据用于动画
        orbits[sat] = (x, y, z)
        
        # 绘制完整轨道
        ax.plot(x, y, z, 
                color=sat_colors[sat], 
                alpha=0.5, 
                linewidth=1,
                label=f'Sat {sat}')
    
    # 5. 添加当前卫星位置标记
    current_positions = {}
    for sat in unique_sats:
        sat_df = df[df['sat_id'] == sat]
        last_point = sat_df.iloc[-1]
        marker = ax.scatter(last_point['x'], last_point['y'], last_point['z'], 
                           color=sat_colors[sat], s=50, edgecolor='black')
        current_positions[sat] = marker
    
    # 6. 设置图表属性
    max_range = max(df[['x', 'y', 'z']].abs().max().max(), earth_radius * 1.5)
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    
    ax.set_title('GNSS Orbit', fontsize=16)
    
    # 添加图例
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # 7. 添加时钟信息
    time_text = ax.text2D(0.05, 0.95, f"Time: {df['time'].iloc[-1]}", 
                          transform=ax.transAxes, fontsize=12)
    
    # 8. 创建动画函数
    def update(frame):
        # 更新所有卫星位置
        for sat in unique_sats:
            x, y, z = orbits[sat]
            current_positions[sat]._offsets3d = ([x[frame]], [y[frame]], [z[frame]])
        
        # 更新时间文本
        time_text.set_text(f"Time: {df['time'].iloc[frame]}")
        
        return list(current_positions.values()) + [time_text]
    
    # 9. 创建动画
    frames = min(len(df[df['sat_id'] == sat]) for sat in unique_sats)
    ani = animation.FuncAnimation(fig, update, frames=frames, 
                                  interval=100, blit=True)
    
    plt.tight_layout()
    plt.show()
    
    return ani

def VisualizeOrbit2d(df):
    df = convert_time_column(df, time_col='time')
    ids = df['sat_id'].unique()
    plt.figure(figsize=(15, 10))
    plt.subplot(311)
    for sat_id in ids:
      x = df[df['sat_id']==sat_id]['x']
      time = df[df['sat_id']==sat_id]['time']
      plt.plot(time, x,label=sat_id+'_x')
    plt.xlabel('Epoch')
    plt.ylabel('X (km)')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.subplot(312)
    for sat_id in ids:
      y = df[df['sat_id']==sat_id]['y']
      plt.plot(y,label=sat_id+'_y')
    plt.xlabel('Epoch')
    plt.ylabel('Y (km)')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.subplot(313)
    for sat_id in ids:
      z = df[df['sat_id']==sat_id]['z']
      plt.plot(z,label=sat_id+'_z')
    plt.xlabel('Epoch')
    plt.ylabel('Z (km)')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
      
def visualize_dyn(df, name, config):
    """
    可视化动力学拟合参数
    
    参数:
    df: DataFrame包含动力学拟合参数，列名:
        ['start_time', 'EstCd', 'EstCr', 'BiasX', 'BiasY', 'BiasZ', 
         'EmpirAccX', 'EmpirAccY', 'EmpirAccZ', 'time']
    """
    df = convert_time_column(df, time_col='time')
    # 创建图表布局
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(10, 18))
    plt.subplot(8,1,1)
    plt.plot(df['time'], df['EstCd'], 'b-', label='Cd')
    plt.xlabel('Epoch')
    plt.ylabel('Cd')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.subplot(8,1,2)
    plt.plot(df['time'], df['EstCr'], 'r-', label='Cr')
    plt.xlabel('Epoch')
    plt.ylabel('Cr')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.subplot(8,1,3)
    plt.plot(df['time'], df['BiasX'], 'b-', label='BiasX')
    plt.xlabel('Epoch')
    plt.ylabel('BiasX')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.subplot(8,1,4)
    plt.plot(df['time'], df['BiasY'], 'g-', label='BiasY')
    plt.xlabel('Epoch')
    plt.ylabel('BiasY')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.subplot(8,1,5)
    plt.plot(df['time'], df['BiasZ'], 'r-', label='BiasZ')
    plt.xlabel('Epoch')
    plt.ylabel('BiasZ')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.subplot(8,1,6)
    plt.plot(df['time'], df['EmpirAccX'], 'b-', label='EmpirAcc_r')
    plt.xlabel('Epoch')
    plt.ylabel('EmpirAcc_r(nm/s²)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.subplot(8,1,7)
    plt.plot(df['time'], df['EmpirAccY'], 'g-', label='EmpirAcc_t')
    plt.xlabel('Epoch')
    plt.ylabel('EmpirAcc_t(nm/s²)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.subplot(8,1,8)
    plt.plot(df['time'], df['EmpirAccZ'], 'r-', label='EmpirAcc_n')
    plt.xlabel('Epoch')
    plt.ylabel('EmpirAcc_n(nm/s²)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if config['savefig']:
      img_name = name+'_dyn.png'
      plt.savefig(os.path.join(config['output_path'],img_name), dpi=300)
    else:
      plt.show()

def visualize_res(data, name, config):
   # res_struct : dict( key: timesystem, value: dict(key: sat_id, value: dict(ele, azi, prefitL, prefitC)) )

   #fig1: prefitL时间序列, fig2: prefitC时间序列
   #fig3: prefitL与仰角关系，fig4: prefitC与仰角关系

  time_list = []
  prefitL_list = []
  prefitC_list = []
  ele_list = []
  for time, sats_data in data.items():
    for sat_id, value_data in sats_data.items():
      time_list.append(time.datetime)
      prefitL_list.append(value_data['prefitL'])
      prefitC_list.append(value_data['prefitC'])
      ele_list.append(value_data['ele'])
  time_list = np.array(time_list)
  prefitL_list = np.array(prefitL_list)
  prefitC_list = np.array(prefitC_list)
  ele_list = np.array(ele_list)

  plot_config = [
     (time_list, prefitL_list, 'Carrier Phase Residuals (prefitL)', 'Time', 'Residual (m)'),
      (time_list, prefitC_list, 'Pseudorange Residuals (prefitC)', 'Time', 'Residual (m)'),
      (ele_list, prefitL_list, 'Carrier Phase Residuals vs Elevation', 'Elevation (°)', 'Residual (m)'),
      (ele_list, prefitC_list, 'Pseudorange Residuals vs Elevation', 'Elevation (°)', 'Residual (m)')
  ]
        
  plt.figure(figsize=(16, 12))
  for i, (x, y, title, xlabel, ylabel) in enumerate(plot_config):
      plt.subplot(2, 2, i+1)
      plt.scatter(x, y, alpha=0.6, s=5)
      plt.title(name+' '+title)
      plt.xlabel(xlabel)
      plt.ylabel(ylabel)
      plt.grid(True, linestyle='--', alpha=0.7)
      if 'Time' in xlabel:
          plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
          plt.text(0.02, 0.95, f'Std: {y.std():.4f} m\nRMSE: {np.sqrt(np.mean(y**2)):.4g} m',
                   transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
      else:
          plt.text(0.02, 0.95, f'Std: {y.std():.4f} m\nRMSE: {np.sqrt(np.mean(y**2)):.4g} m',
                   transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
  if config['savefig']:
    img_name = name+'_res.png'
    plt.savefig(os.path.join(config['output_path'],img_name), dpi=300)
  else:
    plt.show()
  
  result = "prefitL: STD = {:.4f} m, RMSE = {:.4f} m\nprefitC: STD = {:.4f} m, RMSE = {:.4f} m".format(prefitL_list.std(), np.sqrt(np.mean(prefitL_list**2)), prefitC_list.std(), np.sqrt(np.mean(prefitC_list**2)))
  return result



def visualize_res_sky(data, name, config):
  """Plot residual time series plus north-up, clockwise sky plots.

  The polar radius is zenith angle: 0 degrees at the centre and 90 degrees
  at the horizon. Colours show the prefit O-C residual in millimetres.
  """
  time_list, phase_list, code_list, elevation_list, azimuth_list = [], [], [], [], []
  for time, sats_data in data.items():
    for sat_id, values in sats_data.items():
      time_list.append(time.datetime)
      phase_list.append(values['prefitL'])
      code_list.append(values['prefitC'])
      elevation_list.append(values['ele'])
      azimuth_list.append(values['azi'])
  time_array = np.asarray(time_list)
  phase_array = np.asarray(phase_list, dtype=float)
  code_array = np.asarray(code_list, dtype=float)
  elevation_array = np.asarray(elevation_list, dtype=float)
  azimuth_array = np.asarray(azimuth_list, dtype=float)

  fig = plt.figure(figsize=(16, 12))
  for index, (residuals, title) in enumerate([
      (phase_array, 'Carrier Phase Residuals (prefitL)'),
      (code_array, 'Pseudorange Residuals (prefitC)')], start=1):
    axis = fig.add_subplot(2, 2, index)
    axis.scatter(time_array, residuals, alpha=.6, s=5)
    axis.axhline(0, color='k', alpha=.3)
    axis.set_title(name + ' ' + title)
    axis.set_xlabel('Time')
    axis.set_ylabel('Residual (m)')
    axis.grid(True, linestyle='--', alpha=.7)
    # Preserve the daily layout while keeping tick labels legible when this
    # established plotting routine receives a combined multi-day data set.
    if len(time_array) and (max(time_array) - min(time_array)).total_seconds() > 86400:
      # A fixed every-two-day cadence remains readable in the 2-column layout
      # used by this four-panel figure (the automatic locator still crowded it).
      axis.xaxis.set_major_locator(mdates.DayLocator(interval=2))
      axis.xaxis.set_major_formatter(DateFormatter('%m-%d\n%Y'))
    axis.text(.02, .95, f'Std: {residuals.std()*1000:.4f} mm',
              transform=axis.transAxes, va='top', fontsize=10,
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=.5))
  print("prefitL: STD = {:.4f} mm, prefitC: STD = {:.4f} mm".format(
      phase_array.std() * 1000, code_array.std() * 1000))

  def make_sky_plot(axis, residuals, title):
    valid = np.isfinite(azimuth_array) & np.isfinite(elevation_array) & np.isfinite(residuals)
    theta = np.deg2rad(np.mod(azimuth_array[valid], 360.0))
    zenith_angle = np.clip(90.0 - elevation_array[valid], 0.0, 90.0)
    residual_mm = residuals[valid] * 1000.0
    limit = max(float(np.percentile(np.abs(residual_mm), 95.0)), 1e-6) if len(residual_mm) else 1.0
    points = axis.scatter(theta, zenith_angle, c=residual_mm, s=5, cmap='coolwarm',
                          vmin=-limit, vmax=limit, alpha=.82, linewidths=0)
    axis.set_theta_zero_location('N')
    axis.set_theta_direction(-1)
    axis.set_rlim(0, 90)
    axis.set_rgrids([30, 60, 90], labels=['30°', '60°', '90°'], angle=22.5)
    axis.set_title(name + ' ' + title, pad=18)
    colorbar = fig.colorbar(points, ax=axis, pad=.09, shrink=.88)
    colorbar.set_label('O-C residual (mm)')

  make_sky_plot(fig.add_subplot(2, 2, 3, projection='polar'), phase_array, 'Carrier Phase Residual Sky Plot')
  make_sky_plot(fig.add_subplot(2, 2, 4, projection='polar'), code_array, 'Pseudorange Residual Sky Plot')
  fig.tight_layout(pad=2.0, h_pad=3.0, w_pad=2.0)
  if config['savefig']:
    plt.savefig(os.path.join(config['output_path'], name + '_res.png'), dpi=300)
  else:
    plt.show()
  return "prefitL: STD = {:.4f} m, RMSE = {:.4f} m\nprefitC: STD = {:.4f} m, RMSE = {:.4f} m".format(
      phase_array.std(), np.sqrt(np.mean(phase_array**2)), code_array.std(), np.sqrt(np.mean(code_array**2)))


def visualize_res_adopt(df, name, config):
  df = convert_time_column(df, time_col='time')

  sns.set_style("whitegrid")
  
  # 1. 准备数据：提取所有卫星的残差数据，过滤卫星编号为0的数据点
  max_sat = df['sat_num'].max()  # 最大卫星数量
  all_residuals = []
  
  for i in range(1, max_sat + 1):
    sat_id_col = f'sat_id_{i}'
    prefitL_col = f'prefitL_{i}'
    prefitC_col = f'prefitC_{i}'
    
    # 检查列是否存在
    if sat_id_col in df.columns and prefitL_col in df.columns and prefitC_col in df.columns:
      # 提取有效数据，过滤卫星编号为0的数据点
      valid_mask = (df[sat_id_col].notna()) & (df[prefitL_col].notna()) & (df[prefitC_col].notna())
      valid_mask = valid_mask & (df[sat_id_col] != 0)  # 过滤卫星编号为0的数据点
      
      valid_data = df[valid_mask]
      
      for idx, row in valid_data.iterrows():
        all_residuals.append({
          'time': row['time'],
          'sat_id': row[sat_id_col],
          'prefitL': row[prefitL_col],
          'prefitC': row[prefitC_col],
          'elevation': row.get(f'ele_{i}', np.nan),
          'azimuth': row.get(f'azi_{i}', np.nan)
        })
  
  # 创建新的DataFrame
  residuals_df = pd.DataFrame(all_residuals)
  prefitC = np.array(residuals_df['prefitC'])
  prefitL = np.array(residuals_df['prefitL'])
  
  if residuals_df.empty:
    print("No valid residual data found.")
    return
  
  # 获取唯一的卫星ID列表
  unique_sats = residuals_df['sat_id'].unique()
  
  # 创建颜色映射
  colors = plt.cm.tab20(np.linspace(0, 1, len(unique_sats)))
  sat_colors = dict(zip(unique_sats, colors))
  
  # 2. 创建图表布局
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
  
  # 3. 载波相位残差时间序列
  for sat in unique_sats:
      sat_data = residuals_df[residuals_df['sat_id'] == sat]
      ax1.scatter(sat_data['time'], sat_data['prefitL'],
                color=sat_colors[sat], 
                label=f'{sat}', 
                alpha=0.6,
                s=5)
  
  ax1.set_title(name+' Carrier Phase Residuals (prefitL)')
  ax1.set_ylabel('Residual (m)')
  ax1.set_xlabel('Time')
  ax1.text(0.02, 0.95, f'Std: {prefitL.std():.4f} m\nRMSE: {np.sqrt(np.mean(prefitL**2)):.4g} m',
           transform=ax1.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
  ax1.grid(True, linestyle='--', alpha=0.7)
  
  # 添加零参考线
  ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
  
  # 4. 伪距残差时间序列
  for sat in unique_sats:
      sat_data = residuals_df[residuals_df['sat_id'] == sat]
      ax2.scatter(sat_data['time'], sat_data['prefitC'],
                color=sat_colors[sat], 
                label=f'{sat}', 
                alpha=0.6,
                s=5)
  
  ax2.set_title(name+' Pseudorange Residuals (prefitC)')
  ax2.set_ylabel('Residual (m)')
  ax2.set_xlabel('Time')
  ax2.text(0.02, 0.95, f'Std: {prefitC.std():.4f} m\nRMSE: {np.sqrt(np.mean(prefitC**2)):.4g} m',
           transform=ax2.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
  ax2.grid(True, linestyle='--', alpha=0.7)
  
  # 添加零参考线
  ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
  
  # 5. 设置时间轴格式
  date_format = mdates.DateFormatter('%H:%M:%S')
  for ax in [ax1, ax2]:
      ax.xaxis.set_major_formatter(date_format)
      ax.xaxis.set_major_locator(mdates.AutoDateLocator())
      plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
  
  # 6. 添加图例
  handles, labels = ax1.get_legend_handles_labels()
  fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.95), 
              title='Satellite ID', ncol=4)
  
  plt.tight_layout()
  plt.subplots_adjust(top=0.95)
  if config['savefig']:
    img_name = name+'_res.png'
    plt.savefig(os.path.join(config['output_path'],img_name), dpi=300)
  else:
    plt.show()

  plt.figure(figsize=(16, 6))
  plt.subplot(1,2,1)
  plt.hist(residuals_df['prefitL'], bins=50, color='blue', alpha=0.7)
  plt.title(name+' Carrier Phase Residuals Histogram')
  plt.xlabel('Residual (m)')
  plt.ylabel('Frequency')
  plt.grid(True, linestyle='--', alpha=0.7)
  # plt.text(0.5, 0.8, f'std: {prefitL.std():.4f} m\nRMSE: {np.sqrt(np.mean(prefitL**2)):.4g} m')
  plt.subplot(1,2,2)
  plt.hist(residuals_df['prefitC'], bins=150, color='orange', alpha=0.7)
  plt.title(name+' Pseudorange Residuals Histogram')
  plt.xlabel('Residual (m)')
  plt.ylabel('Frequency')
  plt.grid(True, linestyle='--', alpha=0.7)
  # plt.text(0.8, 0.5, f'std: {prefitC.std():.4f} m\nRMSE: {np.sqrt(np.mean(prefitC**2)):.4g} m')
  plt.tight_layout()
  plt.suptitle(name+' Residuals Histogram', fontsize=16)
  plt.subplots_adjust(top=0.92)
  if config['savefig']:
    img_name = name+'_res_hist.png'
    plt.savefig(os.path.join(config['output_path'],img_name), dpi=300)
  else:
    plt.show()
  
  result = config['satellite']+'_'+name + f'  prefitC mean: {prefitC.mean():4f}, std: {prefitC.std():4f}, RMSE: {np.sqrt(np.mean(prefitC**2)):.4g} prefitL mean: {prefitL.mean():4f}, std: {prefitL.std():4f}, RMSE: {np.sqrt(np.mean(prefitL**2)):.4g}'
  return result

def visualize_editing(df, config):

  df = convert_time_column(df, time_col='time')
  sns.set_style("whitegrid")
  plt.figure(figsize=(10, 6))
  plt.plot(df['time'], df['sat_num0'], 'b-', linewidth=1.5, label='Sat Num before Editing')
  plt.plot(df['time'], df['sat_num1'], 'r-', linewidth=1.5, label='Sat Num after Code Editing')
  plt.plot(df['time'], df['sat_num2'], 'g-', linewidth=1.5, label='Sat Num after Phase Editing')
  #统计satnum2中大于等于4的比例
  count_4 = (df['sat_num2'] >= 4).sum()
  total_count = len(df)
  ratio_4 = count_4 / total_count * 100
  count_5 = (df['sat_num2'] >= 5).sum()
  ratio_5 = count_5 / total_count * 100
  count_6 = (df['sat_num2'] >= 6).sum()
  ratio_6 = count_6 / total_count * 100
  plt.text(0.02, 0.95, f'Sat Num >=4 : {ratio_4:.2f} %\nSat Num >=5 : {ratio_5:.2f} %\nSat Num >=6 : {ratio_6:.2f} %',
           transform=plt.gca().transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
  plt.xlabel('Time')
  plt.ylabel('Sat Num')
  plt.title('Satellite Number During Data Editing', fontsize=14)
  plt.grid(True, linestyle='--', alpha=0.7)
  plt.legend()
  plt.tight_layout()
  if config['savefig']:
    plt.savefig(os.path.join(config['output_path'],'data_editing.png'), dpi=300)
  else:
    plt.show()

def visualize_3d_error(error_df, config):
  plt.plot(error_df['time'], error_df['x_err'], label='X error (m)')
  plt.plot(error_df['time'], error_df['y_err'], label='Y error (m)')
  plt.plot(error_df['time'], error_df['z_err'], label='Z error (m)')
  plt.title(config['plot_title'])
  plt.xlabel('Time')
  plt.ylabel('Position Error (m)')
  plt.legend()
  plt.grid()
  plt.tight_layout()
  if config['savefig']:
    plt.savefig(config['output_path'])
