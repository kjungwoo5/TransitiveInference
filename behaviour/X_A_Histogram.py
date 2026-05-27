import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl

import matplotlib.pyplot as plt
from tqdm import tqdm

from xdetectioncore.behaviour import get_main_sess_td_df
from xdetectioncore.paths import posix_from_win

STAGE = 3

ANIMALS = [

]
DAYS = [

]

SESSION_PATH = r"X:\Dammy\Xdetection_mouse_hf_test\session_topology_transitiveinference_pilot.csv"
HOME_DIR = r"C:\bonsai\data\JungWoo"

def load_aggregate_td_df(session_topology: pd.DataFrame, home_dir:Path, td_df_query=None) -> pd.DataFrame:

    session_topology.dropna(how='any', inplace=True)

    td_path_pattern = '<name>/TrialData'

    abs_td_paths = session_topology['tdata_file'].to_list()

    sessnames = [Path(td_info.replace('_TrialData', '')).stem
                 for td_info in abs_td_paths]

    for abs_td_path in abs_td_paths:
        if not os.path.exists(abs_td_path):
            print(f'Warning: td file not found: {abs_td_path}')
    
    td_dfs = {sessname:pd.read_csv(abs_td_path) for sessname, abs_td_path in zip(sessnames,abs_td_paths) if os.path.exists(abs_td_path)}
        
    td_df = pd.concat(list(td_dfs.values()), keys=td_dfs.keys(), names=['session_name'], axis=0)
    if td_df_query:
        td_df = td_df.query(td_df_query)
    return td_df

df = load_aggregate_td_df(pd.read_csv(SESSION_PATH), Path(HOME_DIR), 'Stage == @STAGE')

if DAYS:
    df = df[df['date'].isin(DAYS)]

df.reset_index(names=['Session_Full', 'Animal_Name', 'Date', 'Session', 'Trial_Number'], inplace=True)

df['Pattern1_Time'] = pd.to_timedelta(df['Pattern1_Time']).dt.total_seconds()
df['Gap_Time'] = pd.to_timedelta(df['Gap_Time']).dt.total_seconds()

df['prev_gap_time'] = df.groupby('Session_Full')['Gap_Time'].shift(1)

df = df[df['Pattern1_Time']>0]

df['X_A'] = df['Pattern1_Time'] - df['prev_gap_time']
print(df['X_A'])
df = df[(df['X_A'] >= 0) & (df['X_A'] <= 30)]

df['X_A'].plot.hist(bins=50, xlim=(-5,30), title=f'Histogram of Time between X and Pattern onset in stage {STAGE}')
plt.show()


