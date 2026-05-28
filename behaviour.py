import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import os

import matplotlib.pyplot as plt
from tqdm import tqdm

# Plot reaction time using a dataframe containing trial data with session-level multiindex 
# (session_name, animal_name, date, session, trial_number) and columns including 'RewardTone_Time' and 'Gap_Time'
def plot_reaction_time(td_df: pd.DataFrame, output_path: Path, days_list = None, td_df_query=None, show_plots = False):
    df = td_df.copy()
    if days_list:
        print(f'Selecting for the days: {days_list}')
        df = df[df['date'].isin(days_list)]
    if td_df_query:
        df = df.query(td_df_query)

    df['RewardTone_Time'] = pd.to_timedelta(df['RewardTone_Time']).dt.total_seconds()
    df['Gap_Time'] = pd.to_timedelta(df['Gap_Time']).dt.total_seconds()

    df = df[df['RewardTone_Time'] > 0]

    df['Reaction_Time'] = df['RewardTone_Time'] - df['Gap_Time']
    df = df[(df['Reaction_Time'] >= 0) & (df['Reaction_Time'] <= 1)]

    for session in tqdm(df['session_name'].unique()):
        single_df = df[df['session_name'] == session]
        single_df = single_df.reset_index()
        single_df['Time (min)'] = (single_df['index'] - single_df['index'][0]) / 60.0
        single_df['Reaction_Time'].plot(x='Time (min)', title=f'Reaction time over time for {session}',
                                        xlabel='Time (min)', ylabel='Reaction Time (s)', ylim=(0,1))
        fig = plt.gcf()
        if show_plots:
            plt.show()
        fig.savefig(output_path / fr'Reaction Times\{session}_reactiontimes.png')


# Plot time between X and subsequent pattern onset using a dataframe containing trial data with session-level multiindex 
# (session_name, animal_name, date, session, trial_number) and columns including 'RewardTone_Time' and 'Gap_Time'
def plot_X_A_time(td_df: pd.DataFrame, stage : int, output_path: Path, days_list = None, td_df_query=None, show_plots = False):
    df = td_df.copy()
    if days_list:
        print(f'Selecting for the days: {days_list}')
        df = df[df['date'].isin(days_list)]
    if td_df_query:
        df = df.query(td_df_query)
    
    df['Pattern1_Time'] = pd.to_timedelta(df['Pattern1_Time']).dt.total_seconds()
    df['Gap_Time'] = pd.to_timedelta(df['Gap_Time']).dt.total_seconds()

    df['prev_gap_time'] = df.groupby('session_name')['Gap_Time'].shift(1)

    df = df[df['Pattern1_Time']>0]

    df['X_A'] = df['Pattern1_Time'] - df['prev_gap_time']
    df = df[(df['X_A'] >= 0) & (df['X_A'] <= 30)]

    df['X_A'].plot.hist(bins=50, xlim=(-5,30), title=f'Histogram of time between X and next pattern onset in stage {stage}')
    
    fig = plt.gcf()
    if show_plots:
        plt.show()
    fig.savefig(output_path / fr'X_A Times\X_A_Times_Stage{stage}.png')

