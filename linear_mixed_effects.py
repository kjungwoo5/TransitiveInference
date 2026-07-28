import statsmodels.formula.api as smf

import json

from pathlib import Path
import sys
sys.path.append('../')
sys.path.append(str(Path(__file__).parent.parent.parent))
from Analysis.XdetectionCore.xdetectioncore.decoding.decoding_funcs import Decoder
from joblib import Parallel, delayed

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import permutations
from sklearn.decomposition import PCA
from sklearn import svm

import behaviour as tfb
import data_io as tfio
import pupillometry as tfp
from pupillometry import PupilPlotter

SESSION_PATH = Path(r"X:\Dammy\Xdetection_mouse_hf_test\session_topology_transitive_inference_full.csv")
HOME_PATH = Path(r"C:\bonsai\data\JungWoo")
OUTPUT_PATH = Path(r"C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\Outputs")
PARQUET_DIR = Path(r'X:\Dammy\mouse_pupillometry\pickles\trans_inf_test_90Hz_hpass00_lpass0')
HARP_DIR = Path(r'X:\Dammy\harpbins')

STAGE = 5
    
pupil_df = tfio.load_aggregate_pupil_df(SESSION_PATH, STAGE, PARQUET_DIR)
harp_df = tfio.load_aggregate_harp_df(SESSION_PATH, STAGE, HARP_DIR)
td_df = tfio.load_aggregate_trial_data(SESSION_PATH, HOME_PATH)


harp_filtered = tfio.filter_harp_by_successful_trials(harp_df, td_df, print_trial_lengths=False)


for animal in ['JK01', 'JK02', 'JK03', 'JK04']:
    plotter = PupilPlotter(pupil_df, harp_filtered, STAGE, 'testing', OUTPUT_PATH, [animal])
    plotter.align_pupil_by_session(filter=True)
    
    
    window_size = 0.5
    offset = 0.63
    
    accuracies_by_window = {}

    print(f'\nwindow size: {window_size}')
    pip_df = plotter.prep_for_decoding(window_size=window_size, tmax= offset)
    
    pip_df.dropna(inplace=True)
    
    pip_df = pip_df[pip_df['stimulus_id'] != 'X']
    pip_df = pip_df[pip_df['stimulus_id'] != 'GHAB']
    pip_df = pip_df[pip_df['stimulus_id'] != 'ABGH']
    
    pip_df['first_tone'] = pip_df['stimulus_id'].str[0]
    pip_df['second_tone'] = pip_df['stimulus_id'].str[1]
    pip_df['third_tone'] = pip_df['stimulus_id'].str[2]
    pip_df['fourth_tone'] = pip_df['stimulus_id'].str[3]
    
    # Sample by minimum number of occurrences for all stimuli 
    min_stimulus_id = pip_df['stimulus_id'].value_counts().min()
    subsampled_stimuli = pip_df.groupby('stimulus_id').sample(min_stimulus_id)
    subsampled_stimuli.reset_index(inplace=True)
    
    # Sample by minimum number of occurrences for first tones
    min_first_tone = pip_df['first_tone'].value_counts().min()
    subsampled_first = pip_df.groupby('first_tone').sample(min_first_tone)
    subsampled_first.reset_index(inplace=True)
    
    # Sample by minimum number of occurrences for second tones
    min_second_tone = pip_df['second_tone'].value_counts().min()
    subsampled_second = pip_df.groupby('second_tone').sample(min_second_tone)
    subsampled_second.reset_index(inplace=True)
    
    # Sample by minimum number of occurrences for third tones
    min_third_tone = pip_df['third_tone'].value_counts().min()
    subsampled_third = pip_df.groupby('third_tone').sample(min_third_tone)
    subsampled_third.reset_index(inplace=True)
    
    # Sample by minimum number of occurrences for fourth tones
    min_fourth_tone = pip_df['fourth_tone'].value_counts().min()
    subsampled_fourth = pip_df.groupby('fourth_tone').sample(min_fourth_tone)
    subsampled_fourth.reset_index(inplace=True)
    
    print(subsampled_first['first_tone'].value_counts())
    print(subsampled_second['second_tone'].value_counts())
    print(subsampled_third['third_tone'].value_counts())
    print(subsampled_fourth['fourth_tone'].value_counts())
    print(subsampled_stimuli['stimulus_id'].value_counts())

# 2. FIT MIXED-EFFECTS INTERACTION MODEL
    # Formula: Distance = intercept + type + position + (type * position)
    formula = "cosine_distance ~ C(deviant_type) * C(pattern_position)"
    model = smf.mixedlm(formula, data=df, groups=df["session_id"])
    results = model.fit()