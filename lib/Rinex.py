# gnssbox        : The most complete GNSS Python toolkit ever
# readObs        : Read Obs
# Author         : Chang Chuntao chuntaochang@whu.edu.cn
# Copyright(C)   : The GNSS Center, Wuhan University
# Creation Date  : 2022.07.14
# Latest Version : 2023.04.27

"""
RINEX观测数据处理模块 (Rinex.py)
---------------------------------------------------------------------
功能描述:
    本模块专门用于处理RINEX (Receiver Independent Exchange Format) 格式的GNSS观测数据。
    实现了对RINEX 2.x 和 3.x 格式文件的自动识别、读取、解析以及格式转换功能。
    主要用于FAST软件或相关GNSS数据处理任务中。

文件历史:
    2022.07.14 - 创建 (Chang Chuntao)
    2023.04.27 - 最新版本更新
    2025-10-29 - v 0.1 FAST中的RINEX观测数据读取模块说明添加

核心函数说明:
    1. 读取功能:
        - readObs(obsFile, ...): 智能读取入口，自动判断版本(2.x/3.x)并返回观测数据。
        - readObsHead(obsFile, ...): 读取文件头信息。
        - readObs2 / readObs2Head: 针对RINEX 2.x格式的具体实现。
        - readObs3 / readObs3Head: 针对RINEX 3.x格式的具体实现。
    
    2. 写入与转换:
        - writeObs(...): 将观测数据写入为RINEX 3.05格式文件。
        - convert_rinex2_to_rinex3(...): 提供将2.x格式文件转换为3.x格式的便捷函数。
        - reWriteObs(...): 对观测数据进行筛选或修改后重新写入。

    3. 工具函数:
        - is_obs(file): 检查文件是否为有效的RINEX观测文件。
        - get_obs_in_path(path): 遍历目录下所有RINEX观测文件。
        - getIntervalInRinex3(file): 计算采样间隔。
        - readRinex*LineTime: 解析各版本时间格式。

详细数据结构说明:
----------------
1. obsHead (字典 Dict)
   存储RINEX文件的头部元数据信息。
   
   Structure:
   {
       'version': float,       # RINEX版本号 (e.g., 2.11, 3.02)
       'PGM': str,             # 生成程序的名称
       'RUN': str,             # 运行机构或创建者
       'MARKER NAME': str,     # 测站名称
       'MARKER NUMBER': str,   # 测站编号
       'Receiver': str,        # 接收机序列号
       'Receiver Type': str,   # 接收机型号
       'Antenna': str,         # 天线序列号
       'Antenna Type': str,    # 天线型号
       'Agency': str,          # 观测机构
       'Approx Position': [x, y, z],  # 概略坐标 (ECEF, unit: m)
       'Antenna Delta': [h, e, n],    # 天线高 (HEN, unit: m)
       'interval': int/float,  # 观测间隔 (seconds)
       'Leap Seconds': int,    # 跳秒数
       'prn': list,            # 包含的卫星PRN列表 (e.g., ['G01', 'C02', ...])
       'OBS TYPES': dict,      # 观测值类型定义 (按卫星系统分类)
                               # Key: 系统标识 ('G', 'C', 'E', 'R' ...)
                               # Value: 观测类型列表 (e.g., ['C1C', 'L1C', 'D1C'...])
                               # 注意：RINEX 2.x的观测类型会被自动映射/转换为RINEX 3.x的标准格式代码。
       'original_OBS_TYPES': list # 原始头部中的观测类型记录
   }
   
2. obsData (字典 Dict)
   存储解析后的观测数据体。
   这是一个多层嵌套字典，层级结构如下：
   
   obsData[EpochTime][PRN][ObsType] = Value
   
   - Level 1 Key: EpochTime (datetime.datetime)
     观测历元时间，Python datetime对象。
     
   - Level 2 Key: PRN (str)
     卫星编号，如 'G01', 'C02', 'E05'。
     
   - Level 3 Key: ObsType (str)
     观测值类型，对应 obsHead['OBS TYPES'] 中的字符串 (如 'C1C', 'L1C')。
     对于RINEX 3.x文件，可能还会包含LLI (Loss of Lock Indicator) 信息，
     Key格式为 观测类型 + 'LLI' (如 'L1CLLI')。
     
   - Value: float or int or None
     具体的观测数值。如果数据缺失或解析失败，通常为 None。
     LLI值为整数。

依赖:
    - datetime, time, os
    - PyQt5 (可选，用于UI进度条显示)
"""


def readRinex3LineTime(line):
    '''
    读取rinex3.x文件时间转为datetime   
        by ChangChuntao -> 2022.07.14
    适应非整秒
        by ChangChuntao -> 2023.09.04
    '''
    import datetime
    format_string = "> %Y %m %d %H %M %S.%f"
    try:
        lineDatetime = datetime.datetime.strptime(line[:28], format_string)
    except ValueError:
        return None
    return lineDatetime

def readRinex2LineTime(line):
    '''
    读取rinex2.x文件时间转为datetime    by ChangChuntao -> 2022.07.14
    '''
    import datetime
    line = str(line).split()
    year = int(line[0])
    month = int(line[1])
    day = int(line[2])
    hour = int(line[3])
    minute = int(line[4])
    second = round(float(line[5]))
    if year < 80:
        year += 2000
    else:
        year += 1900
    lineDatetime = datetime.datetime(year, month, day, hour, minute)
    lineDatetime += datetime.timedelta(seconds=second)
    return lineDatetime



def readObs2Head(obsFile, needSatList = True, bar = None):
    '''
    读取rinex2.x文件的文件头    by ChangChuntao -> 2022.09.25
    '''
    
    # bar for QT
    if bar is not None:
        from PyQt5.QtWidgets import QApplication
        bar.status.showMessage("Reading Head...")
        QApplication.processEvents()

    obsFileOpen = open(obsFile, 'r')
    obsFileLines = obsFileOpen.readlines()
    obsFileOpen.close()
    gnssSysList = ['C', 'R', 'G', 'E', 'I', 'J', 'S', 'W', '0']
    obsHead = {'version': None,
                'PGM': 'gnssbox',
                'RUN': 'WHU',
                'MARKER NAME': None,
                'MARKER NUMBER': None,
                'Receiver': None,
                'Receiver Type': None,
                'Antenna': None,
                'Antenna Type': None,
                'Agency': None,
                'Approx Position': [None, None, None],
                'Antenna Delta': [None, None, None],
                'interval': None,
                'Leap Seconds': None,
                'epoch': [],
                'prn': [],
                'OBS TYPES': None,
                'original_OBS_TYPES': []}
    satList = []
    endHeadLineNum = 0
    for line in obsFileLines:
        endHeadLineNum += 1
        if 'END OF HEADER' in line:
            endHeadLineNum += 1
            break
    satNum = 0
    if needSatList:
        for lineIndex in range(len(obsFileLines[endHeadLineNum:])):
            nowLineIndex = endHeadLineNum-1+lineIndex
            nowLine = obsFileLines[nowLineIndex]
            satListInEpoch = []
            try:
                readRinex2LineTime(nowLine)
                satNum = int(nowLine[30:32])
                while len(satListInEpoch) < satNum:
                    for prn in [nowLine[32:-1][i:i+3] for i in range(0, len(nowLine[32:-1]), 3)]:
                        if prn == '   ':
                            break
                        if prn[0] == ' ':
                            prn = 'G' + prn[1:]
                        satListInEpoch.append(prn)
                    nowLineIndex += 1
                    nowLine = obsFileLines[nowLineIndex]
                for prn in satListInEpoch:
                    if prn not in satList:
                        satList.append(prn)
            except:
                pass
    obsHead['prn'] = satList
    obs = {}
    for line in obsFileLines:
        if 'RINEX VERSION / TYPE' in line:
            obsHead['version'] = float(line.split()[0])
        elif 'PGM / RUN BY / DATE' in line:
            obsHead['PGM'] = line[:20].strip()
            obsHead['RUN'] = line[20:40].strip()
        elif 'MARKER NAME' in line:
            obsHead['MARKER NAME'] = line[:20].strip()
        elif 'MARKER NUMBER' in line:
            obsHead['MARKER NUMBER'] = line[:20].strip()
        elif 'OBSERVER / AGENCY' in line:
            obsHead['Agency'] = line[20:40].strip()
        elif 'REC # / TYPE / VERS' in line:
            obsHead['Receiver'] = line[:20].strip()
            obsHead['Receiver Type'] = line[20:40].strip()
        elif 'ANT # / TYPE' in line:
            obsHead['Antenna'] = line[:20].strip()
            obsHead['Antenna Type'] = line[20:40].strip()
        elif 'APPROX POSITION XYZ' in line:
            x = float(line.split()[0])
            y = float(line.split()[1])
            z = float(line.split()[2])
            obsHead['Approx Position'] =[x, y, z]
        elif 'ANTENNA: DELTA H/E/N' in line:
            h = float(line.split()[0])
            e = float(line.split()[1])
            n = float(line.split()[2])
            obsHead['Antenna Delta'] = [h, e, n]
        elif 'INTERVAL' in line:
            obsHead['interval'] = int(float(line.split()[0]))
        elif 'LEAP SECONDS' in line:
            obsHead['Leap Seconds'] = int(line.split()[0])
        elif '# / TYPES OF OBSERV' in line:
            gnssSys = line[0]
            if gnssSys == ' ':
                gnssSys = 'G'
            if gnssSys in gnssSysList:
                if gnssSys in obs:
                    obs[gnssSys]  += line[10:60].split()
                else:
                    obs[gnssSys] = line[10:60].split()
            else:
                obs[gnssSys] += line[10:60].split()


    for prn in satList:
        if prn[0] not in obs:
            obs[prn[0]] = obs['G']
    
    for gSys in obs:
        newBandList = []
        for bandIndex in range(len(obs[gSys])):
            newBandList.append(obsType2to3(gSys, obs[gSys][bandIndex]))
            obsHead['original_OBS_TYPES'].append(obs[gSys][bandIndex])
        obs[gSys] = newBandList

    obsHead['OBS TYPES'] = obs
    return obsHead

def readObs3Head(obsFile, needSatList = False, bar = None):
    '''
    读取rinex3.x文件的文件头    by ChangChuntao -> 2022.09.24
    '''
    
    if bar is not None:
        from PyQt5.QtWidgets import QApplication
        bar.status.showMessage("Reading Head...")
        QApplication.processEvents()

    obsHead = {'version': None,
                'PGM': 'gnssbox',
                'RUN': 'WHU',
                'MARKER NAME': None,
                'MARKER NUMBER': None,
                'Receiver': None,
                'Receiver Type': None,
                'Antenna': None,
                'Antenna Type': None,
                'Agency': None,
                'Approx Position': [None, None, None],
                'Antenna Delta': [None, None, None],
                'interval': None,
                'Leap Seconds': None,
                'epoch': [],
                'prn': [],
                'OBS TYPES': None,
                'original_OBS_TYPES': []}
    gnssSysList = ['C', 'R', 'G', 'E', 'I', 'J', 'S', 'W', '0', 'X']
    obs = {}
    obsFileOpen = open(obsFile, 'r')
    line = obsFileOpen.readline()
    readBegin = False
    satList = []
    while line != '':
        if needSatList and readBegin:
            if '>' not in line:
                sat = line[:3]
                gnssSys = line[0]
                if gnssSys in ['W', '0', 'X', 'L', '5', '1']:
                    gnssSys = 'W'
                    sat = gnssSys + sat[1:]
                if sat not in satList:
                    satList.append(sat)
        if 'RINEX VERSION / TYPE' in line:
            obsHead['version'] = float(line.split()[0])
        elif 'PGM / RUN BY / DATE' in line:
            obsHead['PGM'] = line[:20].strip()
            obsHead['RUN'] = line[20:40].strip()
        elif 'MARKER NAME' in line:
            obsHead['MARKER NAME'] = line[:20].strip()
        elif 'MARKER NUMBER' in line:
            obsHead['MARKER NUMBER'] = line[:20].strip()
        elif 'OBSERVER / AGENCY' in line:
            obsHead['Agency'] = line[20:40].strip()
        elif 'REC # / TYPE / VERS' in line:
            obsHead['Receiver'] = line[:20].strip()
            obsHead['Receiver Type'] = line[20:40].strip()
        elif 'ANT # / TYPE' in line:
            obsHead['Antenna'] = line[:20].strip()
            obsHead['Antenna Type'] = line[20:40].strip()
        elif 'APPROX POSITION XYZ' in line:
            x = float(line.split()[0])
            y = float(line.split()[1])
            z = float(line.split()[2])
            obsHead['Approx Position'] =[x, y, z]
        elif 'ANTENNA: DELTA H/E/N' in line:
            h = float(line.split()[0])
            e = float(line.split()[1])
            n = float(line.split()[2])
            obsHead['Antenna Delta'] = [h, e, n]
        elif 'INTERVAL' in line:
            obsHead['interval'] = int(float(line.split()[0]))
        elif 'LEAP SECONDS' in line:
            obsHead['Leap Seconds'] = int(line.split()[0])
        elif 'SYS / # / OBS TYPES' in line:
            if line[0] in gnssSysList:
                gnssSys = line[0]
                if gnssSys in ['W', '0', 'X', 'L', '5', '1']:
                    gnssSys = 'W'
                if gnssSys in obs:
                    obs[gnssSys] = obs[gnssSys] + line[7:60].split()
                else:
                    obs[gnssSys] = line[7:60].split()
            else:
                obs[gnssSys] += line[7:60].split()
        elif 'END OF HEADER' in line:
            if needSatList:
                readBegin = True
            else:
                break
        line = obsFileOpen.readline()
    obsHead['prn'] = satList
    obsFileOpen.close()
    
    for gSys in obs:
        newBandList = []
        for bandIndex in range(len(obs[gSys])):
            band = obs[gSys][bandIndex]
            if len(band) == 2:
                obsHead['original_OBS_TYPES'].append(band)
                newBandList.append(obsType2to3(gSys, band))
            else:
                newBandList.append(band)
        obs[gSys] = newBandList

    obsHead['OBS TYPES'] = obs
    return obsHead


def readObs2(obsFile, obsHead=None, bar = None):
    '''
    读取rinex2.x文件的文件体        by ChangChuntao -> 2022.09.24
    存储方式 -> obsData[epoch][prn][band] -> band value
    '''
    import time
    start_time = time.time()
    
    # bar for QT
    if bar is not None:
        from PyQt5.QtWidgets import QApplication
        QApplication.processEvents()

    if obsHead is None:
        obsHead = readObs2Head(obsFile, needSatList=True)
    obsFileOpen = open(obsFile, 'r')
    obsFileLines = obsFileOpen.readlines()
    obsFileOpen.close()

    obsData = {}
    endHeadLineNum = 0
    for line in obsFileLines:
        endHeadLineNum += 1
        if 'END OF HEADER' in line:
            break
        
    isHeadLine = True
    isHead2Line = False
    satNum = 0
    bandIndex = 0
    nowSatNum = 0
    for lineIndex in range(len(obsFileLines[endHeadLineNum:])):
        line = obsFileLines[endHeadLineNum+lineIndex]
        
        if bar is not None:
            if lineIndex / len(obsFileLines) * 100 - int(lineIndex / len(obsFileLines) * 100) < 1.e-4:
                completed = int(20 * lineIndex / len(obsFileLines))
                remaining = 20 - completed
                barPercent = '=' * completed + '-' * remaining
                percentage = f'{(lineIndex / len(obsFileLines)) * 100:.2f}%'
                bar.status.showMessage("Reading obs   [" + barPercent + '] ' + percentage)
                QApplication.processEvents()
        try:
            if 'COMMENT' in line:
                satIndex = 0
                bandIndex = 0
                continue
            if isHeadLine:
                obsDatetime = readRinex2LineTime(line)
                obsData[obsDatetime] = {}
                satNum = int(line[30:32])
            if isHeadLine or isHead2Line:
                nowSatNum = 0
                for prn in [line[32:-1][i:i+3] for i in range(0, len(line[32:-1]), 3)]:
                    if prn == '   ':
                        break
                    if prn[0] == ' ':
                        prn = 'G' + prn[1:]
                    obsData[obsDatetime][prn] = {}
                    nowSys = prn[0]
                    for band in obsHead['OBS TYPES'][nowSys]:
                        obsData[obsDatetime][prn][band] = None
                if len(obsData[obsDatetime]) < satNum:
                    isHead2Line = True
                    isHeadLine = False
                else:
                    isHead2Line = False
                    isHeadLine = False
            elif not isHeadLine and not isHead2Line:
                nowPrn = list(obsData[obsDatetime])[nowSatNum]
                nowSys = nowPrn[0]
                # obsHead['OBS TYPES'][nowSys]
                line = "{:<80}".format(line[:-1])
                for bandStr in [line[i:i+16] for i in range(0, len(line), 16)]:
                    bandStr = bandStr[:14].strip()
                    try:
                        bandValue = float(bandStr)
                    except:
                        bandValue = None
                    nowBandName = obsHead['OBS TYPES'][nowSys][bandIndex]
                    if bandIndex == len(obsHead['OBS TYPES'][nowSys]) - 1:
                        obsData[obsDatetime][nowPrn][nowBandName] = bandValue
                        bandIndex = 0
                        nowSatNum += 1
                        break 
                    obsData[obsDatetime][nowPrn][nowBandName] = bandValue
                    bandIndex += 1
                if nowPrn == list(obsData[obsDatetime])[-1]:
                    isHeadLine = True
        except:
            isHeadLine = True
            isHead2Line = False
            bandIndex = 0
            satNum = 0
            continue

    end_time = time.time()
    execution_time = end_time - start_time
    print("Read OBS Time : ", execution_time, "s")
    if bar is not None:
        bar.status.showMessage("Read finished [" + 20*'=' + '] ' + "100% Elapsed time " + '%.2f' % (execution_time) + ' s')
        QApplication.processEvents()
    return obsData


def readObs3(obsFile, obsHead = None, bar = None):
    '''
    读取rinex3.x文件的文件体, 存储方式 -> obsDat[epoch][prn][band] -> band value        
        by ChangChuntao -> 2022.09.24
    '''
    import time
    start_time = time.time()

    # bar for QT
    if bar is not None:
        from PyQt5.QtWidgets import QApplication

    if obsHead is None:
        obsHead = readObs3Head(obsFile)
    obsFileOpen = open(obsFile, 'r')
    obsFileLines = obsFileOpen.readlines()
    obsFileOpen.close()

    obsData = dict()
    endHeadLineNum = 0
    for line in obsFileLines:
        endHeadLineNum += 1
        if 'END OF HEADER' in line:
            endHeadLineNum += 1
            break
    obsDatetime = None
    lineIndex = endHeadLineNum - 1
    qual_flag = 0
    for line in obsFileLines[endHeadLineNum-1:]:
        lineIndex += 1
        if bar is not None:
            if lineIndex / len(obsFileLines) * 100 - int(lineIndex / len(obsFileLines) * 100) < 1.e-4:
                completed = int(20 * lineIndex / len(obsFileLines)) - 1
                remaining = 20 - completed
                barPercent = '=' * completed + '>' + '+' * remaining
                percentage = f'{(lineIndex / len(obsFileLines)) * 100:.2f}%'
                bar.status.showMessage("Reading obs   [" + barPercent + '] ' + percentage)
                QApplication.processEvents()
        if line[0] == '>':
            if obsDatetime is not None:
                obsData[obsDatetime].update(prnData)
            obsDatetime = readRinex3LineTime(line)
            if obsDatetime is None:
                qual_flag = 0
                continue
            if obsDatetime not in obsData.keys() and obsDatetime is not None:
                obsData[obsDatetime] = dict()
                prnData = dict()
                qual_flag = 1
        else:
            if qual_flag == 0:
                continue
            prn = line[:4].strip()
            if len(prn) == 4:
                prn = prn[0] + prn[2:].strip()
            gnssSys = line[0]
            if gnssSys in ['W', '0', 'X', 'L', '5', '1']:
                gnssSys = 'W'
                prn = gnssSys + prn[1:]
            nowLineIndex = 3
            if gnssSys not in obsHead['OBS TYPES']:
                continue
            if prn[1] == ' ':
                prn = prn[0] + '0' + prn[2]
            prnData[prn] = dict()
            for band in obsHead['OBS TYPES'][gnssSys]:
                bandStr = line[nowLineIndex:nowLineIndex+14].strip()
                nowLineIndex += 16
                bandValue = None
                if bandStr:
                    try:
                        bandValue = float(bandStr)
                        if bandValue < -999999999.0 or bandValue == 0.0:
                            bandValue = None
                    except:
                        pass
                    
                #     if band[0] == 'L':
                #         if line[nowLineIndex-2].strip():
                #             prnData[prn][band+'LLI'] = int(line[nowLineIndex-2])
                #         else:
                #             prnData[prn][band+'LLI'] = 0
                # else:
                #     if band[0] == 'L': prnData[prn][band+'LLI'] = 0
                prnData[prn][band] = bandValue

    if obsDatetime is not None:
        obsData[obsDatetime].update(prnData)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print("Read OBS Time : ", execution_time, "s")
    if bar is not None:
        bar.status.showMessage("Read finished [" + 20*'=' + '] ' + "100% Elapsed time " + '%.2f' % (execution_time) + ' s')
        QApplication.processEvents()
    return obsData


def readObs(obsFile, obsHead = None, bar=None):
    '''
    读取gnss观测数据文件            by ChangChuntao -> 2022.09.24
    '''
    obsFileOpen = open(obsFile, 'r')
    obsFileLine = obsFileOpen.readline()
    obsFileOpen.close()
    obsVersion = 0.0
    if 'RINEX VERSION / TYPE' in obsFileLine:
        obsVersion = float(obsFileLine.split()[0])
    else:
        return None

    if 2.0 <= obsVersion < 3.0:
        obsData = readObs2(obsFile,obsHead, bar)
    elif 4.0 > obsVersion >= 3.0:
        obsData = readObs3(obsFile,obsHead, bar)
    else:
        print('Higher versions of Rinex are not currently supported!')
        return None
    return obsData

def readObsHead(obsFile, needSatList = False, bar=None):
    '''
    读取gnss观测数据文件头          by ChangChuntao -> 2022.09.24
    '''
    
    obsFileOpen = open(obsFile, 'r')
    obsFileLine = obsFileOpen.readline()
    obsFileOpen.close()
    obsVersion = 0.0
    # for line in obsFileLines:
    if 'RINEX VERSION / TYPE' in obsFileLine:
        obsVersion = float(obsFileLine.split()[0])

    if 2.0 <= obsVersion < 3.0:
        obsHead = readObs2Head(obsFile, needSatList, bar)
    elif 4.0 > obsVersion >= 3.0:
        obsHead = readObs3Head(obsFile, needSatList, bar)
    else:
        print('Higher versions of Rinex are not currently supported!')
        return None
    return obsHead

def is_obs(obs_file):
    '''
    Check whether the file is obs - by chang chuntao 2022.12.03
    '''
    try:
        obs_file_open = open(obs_file , 'r')
        obs_file_line = obs_file_open.readline()
    except:
        return False
    obs_file_open.close()
    if 'RINEX VERSION / TYPE' in obs_file_line and 'NAV' not in obs_file_line and obs_file_line[20] != 'C':
        return True
    else:
        return False

def get_obs_in_path(rinex_path):
    '''
    获取给定路径内所有的obs文件 - by chang chuntao 2022.12.03
    '''
    import os
    output_filelist = []
    for root, dirs, files in os.walk(rinex_path):
        for filename in files:
            abs_filename = os.path.join(root, filename)
            if is_obs(abs_filename):
                output_filelist.append(abs_filename)
    return output_filelist

def getSatInObs3(obsFile):
    '''
    获取obs文件内所有的观测卫星 - by chang chuntao 2022.12.03
    '''
    obsOpen = open(obsFile, 'r')
    obsLines = obsOpen.readlines()
    obsOpen.close()
    readBegin = False
    satList = []
    for line in obsLines:
        if readBegin:
            if '>' not in line:
                line = line.split()
                sat = line[0]
                if sat not in satList:
                    satList.append(sat)
        if 'END OF HEADER' in line:
            readBegin = True
    return satList


def getRinex3SatTime(rinexFile):
    '''
    读取观测文件中卫星对应的时间    by ChangChuntao -> 2022.07.14
    rinexFile         : 观测数据文件
    rinexSatTime      : 存储形式
    {prn1 : [datetime1, datetim2, ..., datetimen],
     prn2 : [datetime1, datetim2, ..., datetimen],
     ...,
     prnn : [datetime1, datetim2, ..., datetimen]}
    '''
    rinexFileLineOpen = open(rinexFile, 'r')
    rinexFileLine = rinexFileLineOpen.readlines()
    rinexFileLineOpen.close()
    satList = []
    rinexSatTime = {}
    endHeadLineNum = 0
    for line in rinexFileLine:
        endHeadLineNum += 1
        if 'END OF HEADER' in line:
            endHeadLineNum += 1
            break
    for line in rinexFileLine[endHeadLineNum:]:
        satPrn = line[:3]
        if satPrn not in satList and line[0] != '>':
            satList.append(satPrn)
            rinexSatTime[satPrn] = []
    timeLine = []
    lineNum = 0
    for line in rinexFileLine:
        if line[0] == '>':
            timeLine.append(lineNum)
        lineNum += 1
    for lineNumIndex in range(0, len(timeLine) - 1):
        nowDateTime = readRinex3LineTime(rinexFileLine[timeLine[lineNumIndex]])
        for line in rinexFileLine[timeLine[lineNumIndex] + 1:timeLine[lineNumIndex + 1]]:
            satPrn = line[:3]
            rinexSatTime[satPrn].append(nowDateTime)
    return rinexSatTime

def getRinex3TimeSeries(rinexFile):
    rinexFileLineOpen = open(rinexFile, 'r')
    rinexFileLine = rinexFileLineOpen.readlines()
    rinexFileLineOpen.close()
    readBegin = False
    timeList = []
    for line in rinexFileLine:
        if readBegin:
            if '>' in line:
                nowTime = readRinex3LineTime(line)
                timeList.append(nowTime)
        if 'END OF HEADER' in line:
            readBegin = True
    def get_list(date):
        return date.timestamp()
    timeList = sorted(timeList, key=lambda date:get_list(date))
    return timeList


def get_obsfile_ST_ET_quick(rinexFile):
    '''
    get clkfile start_datetime & end_datetime quick - by chang chuntao 2023.06.22
    '''
    obs_file_open = open(rinexFile , 'r')
    obs_file_line = obs_file_open.readlines()
    obs_file_open.close()
    for line in obs_file_line:
        if '>' in line[0]:
            start_datetime = readRinex3LineTime(line)
            break
    for line in obs_file_line[-200:]:
        if '>' in line[0]:
            end_line = line
    end_datetime = readRinex3LineTime(end_line)
    return start_datetime, end_datetime

def getIntervalInRinex3(rinexFile):
    rinexFileLineOpen = open(rinexFile, 'r')
    rinexFileLine = rinexFileLineOpen.readlines()
    rinexFileLineOpen.close()
    readBegin = False
    timeList = []
    for line in rinexFileLine:
        if readBegin:
            if '>' in line:
                nowTime = readRinex3LineTime(line)
                timeList.append(nowTime)
        if 'END OF HEADER' in line:
            readBegin = True
    def get_list(date):
        return date.timestamp()
    timeList = sorted(timeList, key=lambda date:get_list(date))
    interval = 999999999999
    for nowTimeIndex in range(len(timeList[1:])):
        intervalTemp = (timeList[nowTimeIndex+1] - timeList[nowTimeIndex]).total_seconds()
        if intervalTemp < interval:
            interval = intervalTemp
    return interval

def obsType2to3(gSys, band):
    if band[0] == 'C' or band[0] == 'L' or band[0] == 'S' or band[0] == 'D':
        if band[1] == '1' or band[1] == '2':
            if gSys == 'G':
                if band == 'L2':
                    band = band[:2] + 'P'
                else:
                    band = band[:2] + 'C'
            else:
                band = band[:2] + 'P'
        else:
            if gSys == 'G':
                band = band[:2] + 'I'
            else:
                band = band[:2] + 'P'
    elif band[0] == 'P':
        band = 'C' + band[1] + 'P'
    else:
        return band
    return band

## Write Obs
def writeObs(obsHead, obsData, obsFile, exclude_bands = []):
    obsWrite = open(obsFile, 'w+')
    obsWrite.write('     3.05           OBSERVATION DATA    M (MIXED)           RINEX VERSION / TYPE\n')
    obsWrite.write('GNSSBOX             CHUNTAO CHANG                           PGM / RUN BY / DATE\n')
    obsWrite.write('GNSS RESEARCH CENTER, WUHAN UNIVERSITY (WHU), P. R. CHINA   COMMENT\n')
    for gnssSys in obsHead['OBS TYPES']:
        bandList = obsHead['OBS TYPES'][gnssSys]
        for exclude_band in exclude_bands:
            if exclude_band in bandList:
                bandList.remove(exclude_band)
        bandNum = len(bandList)
        firstLineBegin = gnssSys + '%5d' % bandNum
        otherLineBegin = 6 * ' '
        line = firstLineBegin
        for i, band in enumerate(bandList):
            i += 1
            line += ' ' + band
            if bandNum > 13 and i != 0 and i % 13 == 0:
                line += (60 - (len(line))) * ' '
                line += 'SYS / # / OBS TYPES\n'
                obsWrite.write(line)
                line = otherLineBegin
            elif bandNum > 13 and i == bandNum:
                line += (60 - (len(line))) * ' '
                line += 'SYS / # / OBS TYPES\n'
                obsWrite.write(line)

            elif bandNum <= 13 and i == bandNum:
                line += (60 - (len(line))) * ' '
                line += 'SYS / # / OBS TYPES\n'
                obsWrite.write(line)

    if obsHead['Receiver'] is not None and obsHead['Receiver Type'] is not None:
        obsWrite.write(obsHead['Receiver'].ljust(20) + obsHead['Receiver Type'].ljust(20) + '                    REC # / TYPE / VERS\n')
    if obsHead['Antenna'] is not None and obsHead['Antenna Type'] is not None:
        obsWrite.write(obsHead['Antenna'].ljust(20) + obsHead['Antenna Type'].ljust(20) + '                    ANT # / TYPE\n')

    firstObsTime = list(obsData)[0]
    lastObsTime = list(obsData)[-1]
    obsWrite.write(firstObsTime.strftime("  %Y    %m    %d    %H    %M    %S.%f     GPS         TIME OF FIRST OBS\n"))
    obsWrite.write(lastObsTime.strftime("  %Y    %m    %d    %H    %M    %S.%f     GPS         TIME OF LAST OBS\n"))
    obsWrite.write('                                                            END OF HEADER\n')
    
    for epoch in obsData:
        obsWrite.write(epoch.strftime("> %Y %m %d %H %M %S.%f0  0 {:02d}\n".format(len(obsData[epoch]))))
        for prn in obsData[epoch]:
            # RINEX 3 observation fields start at column 4 (zero-based index 3)
            # immediately after the three-character satellite identifier.
            prnLine = prn.ljust(3)
            for band in obsData[epoch][prn]:
                if band in exclude_bands:
                    continue
                bandV = obsData[epoch][prn][band]
                if bandV is None:
                    bandV = 0.0
                prnLine += '%14.3f' % bandV + '  '
            prnLine += '\n'
            obsWrite.write(prnLine)

def reWriteObs(obsHead, obsData, checkBand, newObsFile, startDatetime = None, endDatetime = None):
    renewObsData = {}
    for epoch in obsData:
        if startDatetime is not None:
            if epoch < startDatetime:
                continue
        if endDatetime is not None:
            if epoch > endDatetime:
                break
        prnData = {}
        for prn in obsData[epoch]:
            if prn[0] not in list(checkBand):
                continue
            bandAllExist = True
            for band in checkBand[prn[0]]:
                if obsData[epoch][prn][band] is None:
                    bandAllExist = False
            if bandAllExist:
                prnData[prn] = obsData[epoch][prn]
        if prnData != {}:
            renewObsData[epoch] = prnData
    writeObs(obsHead, renewObsData, newObsFile)
    return obsHead, renewObsData


# checkBand = {'G': ['C1C', 'C2W', 'L1C', 'L2W'], 'C': ['C1P', 'C5P', 'L1P', 'L5P'], 'E': ['C1C', 'C5Q', 'L1C', 'L5Q']}

# obsFile = r'E:\Project\NOW\Pos\napos\resource\05071410.23o'
# outFile = obsFile.replace('.23o', '_re.23o')

# from gnssbox.com.ioGnss.readObs import readObs3, readObs3Head
# obsHead = readObs3Head(obsFile)
# obsData = readObs3(obsFile)
# reWriteObs(obsHead, obsData, checkBand=checkBand, newObsFile=outFile)

def convert_rinex2_to_rinex3(input_file, output_file):
    ## input: Rinex2 obs file path, and output Rinex3 obs file path
    if not is_obs(input_file):
        raise ValueError("The input file is not a valid RINEX observation file.")
    obs_head = readObsHead(input_file)
    obs_data = readObs(input_file, obs_head)
    writeObs(obs_head,obs_data,output_file)

# input_file = r"D:\csu\GNSS_MIX\CSUAPPS\CSUPODSApp\BIN\inputdata\Obs\DRO\ANT3\DRO_3_20241227_AfterPCO_Merge3MIX_10-10.rnx"
# output_file = r"D:\csu\GNSS_MIX\CSUAPPS\CSUPODSApp\BIN\inputdata\Obs\DRO\ANT3\MIX_lts.rnx"
# exclude_bands = ['C1P']
# convert_rinex2_to_rinex3(input_file, output_file)


import TimeSystem
import SatID
import numpy as np
import pandas as pd

OBS_PRIORITY = {
    "G": [["1X","1C"], ["2W","2X","2P","5X"]],
    "R": [["1P","1C"], ["2P","2C"]],
    "E": [["1X"],      ["5X","7X","8X"]],
    "J": [["1Z","1X","1C"], ["2X","5X"]],
    "C": [["2I"], ["7I","6I"]],
}

FREQ = {
    "1":1575.42e6, "2":1227.60e6, "5":1176.45e6, "6":1268.52e6, "7":1207.14e6, "8":1191.795e6
}
FREQ = {
    'G':{'1':1575.42e6, '2':1227.60e6, '5':1176.45e6},
    'R':{'1':1602.00e6, '2':1246.00e6},
    'E':{'1':1575.42e6, '5':1176.45e6, '7':1207.14e6, '8':1191.795e6},
    'J':{'1':1575.42e6, '2':1227.60e6, '5':1176.45e6},
    'C':{'2':1561.098e6, '6':1268.52e6, '7':1207.14e6},
}


class Observation:    
    def __init__(self, epoch, sat_id, obs):
        self.timesystem = TimeSystem.TimeSystem().from_datetime(epoch)
        self.sat_id = SatID.SatID(sat_id)
        self.sys = self.sat_id.sys
        self.valid =self.reduceObs(obs)

    def reduceObs(self, obs):

        for k in ["P1","L1","P2","L2",
                  "P1_name","L1_name","P2_name","L2_name"]:
            setattr(self, k, np.nan if "_f" in k or k[0] in "PL" else None)

        def pick_pair(cands):
            for b in cands:
                cname = "C"+b
                lname = "L"+b
                P = obs.get(cname)
                L = obs.get(lname)
                okP = P is not None and P != 0 and not np.isnan(P)
                okL = L is not None and L != 0 and not np.isnan(L)
                if okP and okL:
                    f = FREQ.get(self.sys, {}).get(b[0])
                    return P, L, cname, lname, f
            return np.nan, np.nan, None, None, np.nan

        for i, bands in enumerate(OBS_PRIORITY.get(self.sys, []), 1):

            P, L, Pn, Ln, f = pick_pair(bands)

            setattr(self, f"P{i}", P)
            setattr(self, f"L{i}", L)

            setattr(self, f"f{i}", f)

            setattr(self, f"P{i}_name", Pn)
            setattr(self, f"L{i}_name", Ln)

            setattr(self, f"atx_freq{i}", f'{self.sys}0{Pn[1]}' if Pn is not None else None)
        
        if np.isnan(self.P1) or np.isnan(self.P2) or np.isnan(self.L1) or np.isnan(self.L2):
            return False
        return True

def reduce_rinex(obs_data, sys_list=None, starttime=None, endtime=None, eph_sats=None, exclude_sats=[]):
    if sys_list is None:
        sys_list = ['G', 'C', 'E', 'R', 'I', 'J']
    df = []
    for epoch in obs_data:
        if starttime is not None and epoch < starttime:
            continue
        if endtime is not None and epoch > endtime:
            continue
        for prn in obs_data[epoch]:
            if prn[0] not in sys_list:
                continue
            if prn in exclude_sats:
                continue
            if eph_sats is not None and prn not in eph_sats:
                continue
            data0 = obs_data[epoch][prn]
            data_reduce = {}
            obs = Observation(epoch, prn, data0)
            if obs.valid:
                data_reduce.update(vars(obs))
            else:
                continue
                    
            df.append(data_reduce)
    df = pd.DataFrame(df)
    return df



# obs_file = r"D:\csu\MyTools\time_sync\hkqt26001.rnx"
# obs_head = readObsHead(obs_file)
# obs_data = readObs(obs_file, obs_head)
# df = reduce_rinex(obs_data)
# print(df)
