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
    plt.pause(0.1)
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
        fig.clf()


# Plot time between X and subsequent pattern onset using a dataframe containing trial data with session-level multiindex 
# (session_name, animal_name, date, session, trial_number) and columns including 'RewardTone_Time' and 'Gap_Time'
def plot_X_A_time(td_df: pd.DataFrame, stage : int, output_path: Path, days_list = None, td_df_query=None, show_plots = False):
    df = td_df.copy()
    plt.pause(0.1)
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
    fig.clf()


def plot_learning(td_df: pd.DataFrame, output_path: Path, animals: list, show_plots = False):
    run_length = 5
    
    print(td_df)
    df = td_df[td_df['Stage'] <= 1].copy()
    print(df)
    plt.pause(0.1)
    
    success_rates = {
        'JK01':[],
        'JK02':[],
        'JK03':[],
        'JK04':[],
    }
    
    for session in tqdm(df['session_name'].unique()):
        single_df = df[df['session_name'] == session]

        zero_run_indices = []
        current_zero_run = []
        for idx, outcome in single_df['Trial_Outcome'].items():
            if outcome == 0:
                current_zero_run.append(idx)
            else:
                if len(current_zero_run) > run_length:
                    zero_run_indices.extend(current_zero_run)
                current_zero_run = []

        if len(current_zero_run) > run_length:
            zero_run_indices.extend(current_zero_run)

        if zero_run_indices:
            single_df = single_df.drop(index=zero_run_indices)

        success_rate = single_df['Trial_Outcome'].mean()
        for animal in success_rates.keys():
            if session[:4] == animal:
                success_rates[animal].append(success_rate)
    
    longest_length = 0
    for value in success_rates.values():
        if len(value) > longest_length:
            longest_length = len(value)
    fig, ax = plt.subplots()
    for animal in success_rates.keys():
        success_rates[animal] = [np.nan] * (longest_length - len(success_rates[animal])) + success_rates[animal]
        ax.plot(success_rates[animal], label= animal)
    ax.set_ylim(ymin= 0)
    fig.legend()
    if show_plots:
        plt.show()
    fig.savefig(output_path / fr'Learning\Learning_plot_{run_length}.png')
    fig.clf()