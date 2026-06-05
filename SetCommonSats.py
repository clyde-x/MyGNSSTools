import os 
import numpy as np
import pandas as pd
import Rinex 
import TimeSystem 

def get_common_obs(data_a, data_b):
    """
    遍历两个观测数据，使其只保留他们能共同观测到的卫星
    Traverse two observation data, keeping only the satellites common to both.
    
    Args:
        data_a (dict): Observation data A (obsData structure)
        data_b (dict): Observation data B (obsData structure)
        
    Returns:
        tuple: (new_data_a, new_data_b) containing only common epochs and satellites
    """
    common_epochs = set(data_a.keys()) & set(data_b.keys())
    new_data_a = {}
    new_data_b = {}
    
    # Sort epochs to maintain chronological order
    sorted_epochs = sorted(list(common_epochs))
    
    for epoch in sorted_epochs:
        sats_a = set(data_a[epoch].keys())
        sats_b = set(data_b[epoch].keys())
        common_sats = sats_a & sats_b
        
        if common_sats:
            new_data_a[epoch] = {}
            new_data_b[epoch] = {}
            # Sort satellites for consistent order
            for sat in sorted(list(common_sats)):
                new_data_a[epoch][sat] = data_a[epoch][sat]
                new_data_b[epoch][sat] = data_b[epoch][sat]
                
    return new_data_a, new_data_b

def get_common_sats(data_a, data_b):
    """
    获取两个观测数据的共视卫星列表
    Get the list of common view satellites for two observation data.
    
    Args:
        data_a (dict): Observation data A
        data_b (dict): Observation data B
        
    Returns:
        dict: Keys are epoch times (datetime), Values are lists of common satellite PRNs.
    """
    common_sats_dict = {}
    common_epochs = set(data_a.keys()) & set(data_b.keys())
    sorted_epochs = sorted(list(common_epochs))
    
    for epoch in sorted_epochs:
        sats_a = set(data_a[epoch].keys())
        sats_b = set(data_b[epoch].keys())
        common_sats = sats_a & sats_b
        
        if common_sats:
            common_sats_dict[epoch] = sorted(list(common_sats))
            
    return common_sats_dict

if __name__ == "__main__":
    exclude_bands = ['L1CLLI', 'L2CLLI', 'LGILLI']
    A_path = "D:\\csu\\GNSS_MIX\\CSUAPPS\\CSUPODSApp\\BIN\\TempData\\GFO_C\\GFO_C_PreProcess_Obs_20200901.rnx"
    B_path = "D:\\csu\\GNSS_MIX\\CSUAPPS\\CSUPODSApp\\BIN\\TempData\\GFO_D\\GFO_D_PreProcess_Obs_20200901.rnx"

    print("Reading file A...")
    A_head = Rinex.readObsHead(A_path)
    A_data = Rinex.readObs(A_path, A_head)
    
    print("Reading file B...")
    B_head = Rinex.readObsHead(B_path)
    B_data = Rinex.readObs(B_path, B_head)

    print("Original A epochs:", len(A_data))
    print("Original B epochs:", len(B_data))

    # Test get_common_obs (filter data)
    print("Filtering common satellites (Data)...")
    common_A, common_B = get_common_obs(A_data, B_data)
    print("Common epochs (Data):", len(common_A))
    
    # Test get_common_sats (get list)
    print("Getting common satellites list...")
    common_sats_dict = get_common_sats(A_data, B_data)
    print("Common epochs (List):", len(common_sats_dict))

    if len(common_sats_dict) > 0:
        first_epoch = list(common_sats_dict.keys())[0]
        print(f"Sample common sats at {first_epoch}:")
        print(common_sats_dict[first_epoch])

    A_out_path = A_path.replace(".rnx", "_common.rnx")
    B_out_path = B_path.replace(".rnx", "_common.rnx")
    Rinex.writeObs(A_head, common_A, A_out_path)
    Rinex.writeObs(B_head, common_B, B_out_path)

    print(A_head)
    