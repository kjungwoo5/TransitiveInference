import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import os

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


def load_aggregate_td_df(session_topology: pd.DataFrame, home_dir: Path, td_df_query=None) -> pd.DataFrame:

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
    print(f'Selecting for the days: {DAYS}')
    df = df[df['date'].isin(DAYS)]

df.reset_index(names=['Session_Full', 'Animal_Name', 'Date', 'Session', 'Trial_Number'], inplace=True)

df['RewardTone_Time'] = pd.to_timedelta(df['RewardTone_Time']).dt.total_seconds()
df['Gap_Time'] = pd.to_timedelta(df['Gap_Time']).dt.total_seconds()

df = df[df['RewardTone_Time'] > 0]

df['Reaction_Time'] = df['RewardTone_Time'] - df['Gap_Time']
print(df['Reaction_Time'])
df = df[(df['Reaction_Time'] >= 0) & (df['Reaction_Time'] <= 1)]

for session in df['Session_Full'].unique():
    single_df = df[df['Session_Full'] == session]
    single_df = single_df.reset_index()
    single_df['Time (min)'] = (single_df['index'] - single_df['index'][0]) / 60.0
    single_df['Reaction_Time'].plot(x='Time (min)', title=f'Reaction time over time for {session}',
                                    xlabel='Time (min)', ylabel='Reaction Time (s)', ylim=(0,1))
    fig = plt.gcf()
    plt.show()
    fig.savefig(fr'C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\Outputs\Reaction Times\{session}_reactiontimes.png')


