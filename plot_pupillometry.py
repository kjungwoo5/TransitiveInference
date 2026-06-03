import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np

import behaviour as tfb
import data_io as tfio
import pupillometry as tfp
from pupillometry import PupilPlotter

STAGE = 3

SESSION_PATH = Path(r"X:\Dammy\Xdetection_mouse_hf_test\session_topology_transitiveinference_full.csv")
HOME_PATH = Path(r"C:\bonsai\data\JungWoo")
OUTPUT_PATH = Path(r"C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\Outputs")
PARQUET_DIR = Path(r'X:\Dammy\mouse_pupillometry\pickles\trans_inf_test_90Hz_hpass00_lpass0')
HARP_DIR = Path(r'X:\Dammy\harpbins')

if __name__ == "__main__":
    
    pupil_df = tfio.load_aggregate_pupil_df(SESSION_PATH, None, PARQUET_DIR)
    harp_df = tfio.load_aggregate_harp_df(SESSION_PATH, None, HARP_DIR)

    stage1_dates = ['260526', '260527', '260528', '260529']

    pupil_df1 = pupil_df[pupil_df['session_id'].str.contains('|'.join(stage1_dates))]
    harp_df1 = harp_df[harp_df['session_id'].str.contains('|'.join(stage1_dates))]
    
    s1 = PupilPlotter(pupil_df1, harp_df1, 1, 'testing', OUTPUT_PATH, ['JK01', 'JK02', 'JK03', 'JK04'])
    s3 = PupilPlotter(pupil_df, harp_df, 3, 'testing', OUTPUT_PATH, ['JK01', 'JK02', 'JK03', 'JK04'])
    
    s1.plot_sessionwide_pupil_dilation(show_plot = False)
    s3.plot_sessionwide_pupil_dilation(show_plot = False)
    
    
    '''JK01_stage3_first = PupilPlotter(pupil_df, harp_df, 3, 'first', OUTPUT_PATH, ['JK01'])
    JK01_stage3_first.align_pupil_by_session()
    JK01_stage3_first.plot_overall_distribution(save_figure=False)
    JK01_stage3_first.plot_baseline_sub_aligned_pupil_by_session(save_figure=False)'''
    