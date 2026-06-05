"""
2025-10-29      v 0.1
根据json进行分析和绘图，可以进行优化

"""

import json 
import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt

import Sp3Tools
import TimeSystem
import PlotTool

os.chdir(os.path.dirname(__file__))


def compute_sp3_diff(config):
  root_path = config['root_path']
  sp3_file_1_path = os.path.join(root_path, config['sp3_file_1'])
  sp3_file_2_path = ''
  if config['newest']:
    files = os.listdir(os.path.join(root_path, config['sp3_file2_dir']))
    time0 = os.path.getmtime(sp3_file_1_path)
    for file in files:
      tmp_file = os.path.join(root_path, config['sp3_file2_dir'], file)
      if os.path.getmtime(tmp_file) > time0 and config['POD_type'] in file and tmp_file.endswith('.sp3'):
        sp3_file_2_path = tmp_file
        time0 = os.path.getmtime(tmp_file)
    print('Newest sp3 : ', sp3_file_2_path)
  else:
    sp3_file_2_path = os.path.join(root_path, config['sp3_file_2'])
  output_dir = os.path.join(root_path, config['output_path'])

  sp3_file_1_ = Sp3Tools.SP3(sp3_file_1_path)
  sp3_file_2_ = Sp3Tools.SP3(sp3_file_2_path)
  df1 = sp3_file_1_.asDataFrame()
  df2 = sp3_file_2_.asDataFrame()
  print('SP3 file 2: ', df2.head())

  if sp3_file_1_.header['num_sats'] != sp3_file_2_.header['num_sats']:
    raise ValueError('The number of satellites in the two SP3 files are different. sp3_file_1 has '+str(sp3_file_1_.header['num_sats'])+' satellites, while sp3_file_2 has '+str(sp3_file_2_.header['num_sats'])+' satellites.')
  else:
    df1.drop(columns=['sat_id'],inplace=True)
    df2.drop(columns=['sat_id'],inplace=True)

  if config['set_time']:
    start_time = TimeSystem.TimeSystem()
    start_time.from_str(config['start_time'])
    end_time = TimeSystem.TimeSystem()
    end_time.from_str(config['end_time'])
    df1 = df1[(df1['timesystem'] >= start_time) & (df1['timesystem'] <= end_time)]
    df2 = df2[(df2['timesystem'] >= start_time) & (df2['timesystem'] <= end_time)]
  merged = pd.merge(df1, df2, on=['timesystem'], suffixes=('_1', '_2'))
  merged = merged.dropna(axis=0, how='any')
  if merged.empty:
    raise ValueError('No common epochs found in the two SP3 files.')
  
  merged.loc[merged['clk_1'] > 500, 'clk_1'] = 0
  merged.loc[merged['clk_2'] > 500, 'clk_2'] = 0
  error_df = pd.DataFrame()
  error_df['time'] = merged['timesystem']
  error_df['x_err_mm'] = (merged['x_1'] - merged['x_2'])*1000000
  error_df['y_err_mm'] = (merged['y_1'] - merged['y_2'])*1000000
  error_df['z_err_mm'] = (merged['z_1'] - merged['z_2'])*1000000
  error_df['clk_err_ms'] = (merged['clk_1'] - merged['clk_2'])
  error_df['r_err_mm'] = np.sqrt(error_df['x_err_mm']**2 + error_df['y_err_mm']**2 + error_df['z_err_mm']**2)

  if config['save_csv']:
    error_df.to_csv(os.path.join(output_dir, config['POD_type']+'_diff.csv'), index=False)
  return error_df




## deal with log data
def DataEditingLog(log_path, satellite, date):
  sat_num_df = []
  filepath = os.path.join(log_path, satellite+'_Editing_Log_'+date+'.log')
  with open(filepath, 'r', encoding='GB2312') as file:
    start = False
    for line in file:
      if line.startswith('RejectObsData'):
        start = True
        continue
      if start and not line.startswith('20'):
        break
      if not start:
        continue

      lines = line.split(',')
      time_str = lines[0]
      tmp_time = TimeSystem.TimeSystem()
      tmp_time.from_str(time_str)
      elipse = float(lines[1].split('=')[-1])
      sat_num0 = int(lines[2][-1])
      sat_num1 = int(lines[3][-1])
      sat_num2 = int(lines[4][-2])
      sat_num_df.append([tmp_time, elipse, sat_num0, sat_num1, sat_num2])   
  file.close()
  sat_num_df = pd.DataFrame(sat_num_df, columns=['time', 'elipse', 'sat_num0', 'sat_num1', 'sat_num2'])
  return sat_num_df

def SppLog(log_path, satellite, date):
  spp_log_file = os.path.join(log_path, satellite+'_SPP_Log_'+date+'.log')
  spp_log_list=[]
  with open(spp_log_file, 'r') as f:
    lines = f.readlines()
  spp_log = {}
  for line in lines:
    if line.startswith('YYYY') or line.startswith('NNNN'):
      spp_log_list.append(spp_log)
      spp_log = {}
      spp_log['mode'] = line[:4]
      time_str = line[5:36]
      time = TimeSystem.TimeSystem()
      time.from_str(time_str)
      spp_log['time'] = time
    elif line.startswith('vPrefitC'):
      parts = line.split()
      prefitC = []
      for i in range(1,len(parts)):
        prefitC.append(float(parts[i]))
      spp_log['prefitC'] = prefitC
    elif line.startswith('Num,'):
      parts = line.split()
      spp_log['sat_num'] = int(parts[5][:-1])
      spp_log['precise'] = float(parts[6][:-1])
      spp_log['PDOD'] = float(parts[7][:-1])
      spp_log['RMSPost'] = float(parts[8][:-1])
    elif line.startswith('itrNum'):
      parts = line.split()
      spp_log['itr'] = int(parts[3][:-1])
      spp_log['delta'] = float(parts[4])
    elif line.startswith('Clock'):
      spp_log['clk_offset'] = float(line[13:])
    elif line.startswith('azimuth'):
      azi_line = line[14:]
      parts = azi_line.split()
      azi = []
      for i in range(len(parts)):
        azi.append(float(parts[i]))
      spp_log['azi'] = azi
    elif line.startswith('elevation'):
      ele_line = line[16:]
      parts = ele_line.split()
      ele = []
      for i in range(len(parts)):
        ele.append(float(parts[i]))
      spp_log['ele'] = ele
    
  spp_log_list.append(spp_log)
  spp_log_list = spp_log_list[1:]
  spp_log_df = pd.DataFrame(spp_log_list)
  return spp_log_df

def SppInfo(log_path, satellite, date):
  spp_info_file = os.path.join(log_path, satellite+'_SPP_Info_'+date+'.dat')
  spp_info_df = pd.read_csv(spp_info_file, header=None, sep='\s+', skiprows=1, encoding='gbk')
  spp_info_df.columns = ['year', 'month', 'day', 'hour', 'minute', 'second', 'start_sec', 'sat_num', 'precise', 'PDOD']
  spp_info_df['sat_num'] = spp_info_df['sat_num'].astype(int)
  for i in range(len(spp_info_df)):
    tmp_time = TimeSystem.TimeSystem()
    tmp_time.from_ymdhms(int(spp_info_df.loc[i,'year']), int(spp_info_df.loc[i,'month']), int(spp_info_df.loc[i,'day']),
                          int(spp_info_df.loc[i,'hour']), int(spp_info_df.loc[i,'minute']), int(spp_info_df.loc[i,'second']))
    spp_info_df.loc[i,'time'] = tmp_time 
  spp_info_df.drop(columns=['year', 'month', 'day', 'hour', 'minute', 'second'], inplace=True)
  return spp_info_df

def ResLog(log_path, satellite, date, mode):
  # res_struct : dict( key: timesystem, value: dict(key: sat_id, value: dict(ele, azi, prefitL, prefitC)) )
  # mode: SPP or KIPP or DynFit or RDOD
  mode_str = '_'+mode+'_Res_'
  res_file = os.path.join(log_path, satellite+mode_str+date+'.dat')
  res_total = {}
  with open(res_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
  for line in lines:
    timestr = line[:26]
    time = TimeSystem.TimeSystem()
    time.from_str(timestr)
    sat_num = int(line[42:44])
    line = line[44:]
    line_split = line.split()
    sat_info = {}
    for i in range(sat_num):
      sat_id = line_split[i*5]
      ele = float(line_split[i*5+1])
      azi = float(line_split[i*5+2])
      prefitL = float(line_split[i*5+3])
      prefitC = float(line_split[i*5+4])
      sat_info[sat_id] = {'ele': ele, 'azi': azi, 'prefitL': prefitL, 'prefitC': prefitC}
    res_total[time] = sat_info

  return res_total

def DynLog(log_path, satellite, date, mode):
  # mode: DynFit or RDOD
  mode_str = '_'+mode+'_Dyn_'
  dyn_dat_path = os.path.join(log_path, satellite+mode_str+date+'.dat')
  dyn_dat_df = pd.read_csv(dyn_dat_path, header=None, sep='\t')
  columns = ['year', 'month', 'day', 'hour', 'minute', 'second','start_time', 'EstCd', 'EstCr', 'BiasX', 'BiasY', 'BiasZ', 'EmpirAccX', 'EmpirAccY', 'EmpirAccZ', 'na']
  dyn_dat_df.columns = columns
  for i in range(len(dyn_dat_df)):
    tmp_time = TimeSystem.TimeSystem()
    tmp_time.from_ymdhms(int(dyn_dat_df.loc[i,'year']), int(dyn_dat_df.loc[i,'month']), int(dyn_dat_df.loc[i,'day']),
                          int(dyn_dat_df.loc[i,'hour']), int(dyn_dat_df.loc[i,'minute']), int(dyn_dat_df.loc[i,'second']))
    dyn_dat_df.loc[i,'time'] = tmp_time
  dyn_dat_df.drop(columns=['year', 'month', 'day', 'hour', 'minute', 'second','na'], inplace=True)
  return dyn_dat_df

def PlotLog(config):
  if not config["enable"]:
    return
  if config["if_log"]:
    log_path = config['plot_log_path']
    if os.path.exists(log_path) is False:
      os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(config['plot_log_path'], 'a')
    log_file.write('================= Plot Log =================\n')
    log_file.write('Date: '+config['date']+'  Satellite: '+config['satellite']+'\n')
    log_file.write('log time: '+TimeSystem.TimeSystem().current_time().to_str()+'\n')
  log_path = config['log_path']
  date = config['date']
  satellite = config['satellite']

  if config['spp_log']:
    spp_log_df = SppLog(log_path, satellite, date)
    PlotTool.visualize_spp_log(spp_log_df, config)
  if config['spp_info']:
    spp_info_df = SppInfo(log_path, satellite, date)
    PlotTool.visualize_spp_info(spp_info_df, config)
  if config['spp_res']:
    spp_res = ResLog(log_path, satellite, date, 'SPP')
    result = PlotTool.visualize_res(spp_res, 'SPP', config)
    if config["if_log"]:
      log_file.write(result+'\n')
  if config['rdod_res']:
    rdod_res = ResLog(log_path, satellite, date, 'RDOD')
    result = PlotTool.visualize_res(rdod_res, 'RDOD', config)
    if config["if_log"]:
      log_file.write(result+'\n')
  if config['kipp_res']:
    kipp_res = ResLog(log_path, satellite, date, 'KIPP')
    result = PlotTool.visualize_res(kipp_res, 'KIPP', config)
    if config["if_log"]:
      log_file.write(result+'\n')
  if config['dynfit_dyn']:  
    dynfit_dyn_df = DynLog(log_path, satellite, date, 'DynFit')
    PlotTool.visualize_dyn(dynfit_dyn_df, 'DynFit', config)
  if config['rdod_dyn']:
    rdod_dyn_df = DynLog(log_path, satellite, date, 'RDOD')
    PlotTool.visualize_dyn(rdod_dyn_df, 'RDOD', config)
  if config['data_editing']:
    data_editing_df = DataEditingLog(log_path, satellite, date)
    PlotTool.visualize_editing(data_editing_df, config)

  if config["if_log"]:
    log_file.write('================= End Plot Log =================\n\n')
    log_file.close()

import Rinex
import MathTool
def RinexAnalysis(config):
  rinex_path = config['rinex_path']
  if os.path.exists(rinex_path) is False:
    raise ValueError('The rinex path does not exist.')
  if not Rinex.is_obs(rinex_path):
    raise ValueError('The rinex file is not an observation file.')
  obs_head = Rinex.readObsHead(rinex_path)
  obs_data = Rinex.readObs(rinex_path, obs_head)
  S1C = []
  S2C = []
  for epoch, prn_data in obs_data.items():
    for prn, band_data in prn_data.items():
      if band_data['S1C'] is not None:
        S1C.append(band_data['S1C'])
      if band_data['S2C'] is not None:
        S2C.append(band_data['S2C'])

  S1C = np.array(S1C)
  S2C = np.array(S2C)
  threshold = config['threshold']
  print('S1C less than threshold: '+str(np.sum(S1C < threshold)/len(S1C)))
  print('S2C less than threshold: '+str(np.sum(S2C < threshold)/len(S2C)))
  
def OverlapArcAnalysis(config):
  if not config["enable"]:
    return
  root_path = config['root_path']
  arc1_path = os.path.join(root_path, config['arc1'])
  arc2_path = os.path.join(root_path, config['arc2'])
  if config["auto_set_time"]:
    pass 
  else:
    start_time = TimeSystem.TimeSystem()
    start_time.from_str(config['start_time'])
    end_time = TimeSystem.TimeSystem()
    end_time.from_str(config['end_time'])
  output_dir = os.path.join(root_path, config['output_path'])
  output_file_name = 'overlap_arc_'+start_time.to_digital()+'_'+end_time.to_digital()+'.txt'
  output_file = os.path.join(output_dir, output_file_name)
  arc1 = Sp3Tools.SP3(arc1_path)
  arc1_df = arc1.get_data_df()
  arc2 = Sp3Tools.SP3(arc2_path)
  arc2_df = arc2.get_data_df()
  if arc1.header['num_sats'] != 1 and arc2.header['num_sats'] != 1:
    raise ValueError('The number of satellites in the two SP3 files are more than 1.')
  arc1_df = arc1_df[(arc1_df['time'] >= start_time) & (arc1_df['time'] <= end_time)]
  arc2_df = arc2_df[(arc2_df['time'] >= start_time) & (arc2_df['time'] <= end_time)]
  arc1_df.drop(columns=['sat_id', 'year', 'month', 'day', 'hour', 'minute', 'second'], inplace=True)
  arc2_df.drop(columns=['sat_id', 'year', 'month', 'day', 'hour', 'minute', 'second'], inplace=True)
  merged = pd.merge(arc1_df, arc2_df, on=['time'], suffixes=('_1', '_2'))
  merged = merged.dropna(axis=0, how='any')
  if merged.empty:
    raise ValueError('No common epochs found in the two SP3 files.')
  error_df = pd.DataFrame()
  error_df['time'] = merged['time']
  error_df['x_err'] = (merged['x_1'] - merged['x_2'])*1000
  error_df['y_err'] = (merged['y_1'] - merged['y_2'])*1000
  error_df['z_err'] = (merged['z_1'] - merged['z_2'])*1000
  error_df['clk_err'] = (merged['clk_1'] - merged['clk_2'])
  error_df['r_err'] = np.sqrt(error_df['x_err']**2 + error_df['y_err']**2 + error_df['z_err']**2)
  error_df = PlotTool.convert_time_column(error_df)
  error_3d_config = {
    'plot_title': 'Overlap Arc: '+config['start_time']+' to '+config['end_time'],
    'savefig': config['savefig'],
    'output_path': output_file.replace('.txt', '.png')
  }
  PlotTool.visualize_3d_error(error_df, error_3d_config)    

def cmp_relative(cmp_relative):
  diff1 = compute_sp3_diff(cmp_relative['sp3_diff1'])
  diff2 = compute_sp3_diff(cmp_relative['sp3_diff2'])

  diff = pd.merge(diff1, diff2, on='time', suffixes=('_my', '_jpl'))
  diff = PlotTool.convert_time_column(diff)
  diff['x_err'] = diff['x_err_my'] - diff['x_err_jpl']
  diff['y_err'] = diff['y_err_my'] - diff['y_err_jpl']
  diff['z_err'] = diff['z_err_my'] - diff['z_err_jpl']
  diff.to_csv(cmp_relative['output_path'], index=False)

  print(diff.head())
  PlotTool.visualize_3d_error(diff, cmp_relative['plot_config'])




if __name__ == "__main__":
  json_path = 'AnalysisConfig_DRO.json'
  with open(json_path, 'r') as f:
    configs = json.load(f)

  #RinexAnalysis(configs['rinex'])
  compute_sp3_diff(configs['sp3_diff'])
  PlotTool.plot_error(configs['plot_error'])
  PlotLog(configs['plot_log'])
  # OverlapArcAnalysis(configs['overlap_arc'])
  #cmp_relative(configs['cmp_relative'])