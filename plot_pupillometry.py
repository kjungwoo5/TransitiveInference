import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import permutations

import behaviour as tfb
import data_io as tfio
import pupillometry as tfp
from pupillometry import PupilPlotter

SESSION_PATH = Path(r"X:\Dammy\Xdetection_mouse_hf_test\session_topology_transitive_inference_full.csv")
HOME_PATH = Path(r"C:\bonsai\data\JungWoo")
OUTPUT_PATH = Path(r"C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\Outputs")
PARQUET_DIR = Path(r'X:\Dammy\mouse_pupillometry\pickles\trans_inf_test_90Hz_hpass00_lpass0')
HARP_DIR = Path(r'X:\Dammy\harpbins')

if __name__ == "__main__":
    
    STAGE = 5
    
    pupil_df = tfio.load_aggregate_pupil_df(SESSION_PATH, STAGE, PARQUET_DIR)
    harp_df = tfio.load_aggregate_harp_df(SESSION_PATH, STAGE, HARP_DIR)
    td_df = tfio.load_aggregate_trial_data(SESSION_PATH, HOME_PATH)
    
    harp_filtered = tfio.filter_harp_by_successful_trials(harp_df, td_df, print_trial_lengths=False)
    
    
    # TODO: Plot these again, but without the X tone now. 
    for a in range(1,5):
        plotter = PupilPlotter(pupil_df, harp_filtered, STAGE, 'testing', OUTPUT_PATH, [f'JK0{a}'])
        plotter.align_pupil_by_session(filter=True)
        # plotter.plot_pitch_dependency(save_figure = False, show_means_as_points = True)
        plotter.plot_overall_baseline_sub_aligned_pupil(show_plot = False)
        plotter.plot_stage5_perms(show_plot=False)
    
    
    # plotter = PupilPlotter(pupil_df, harp_filtered, STAGE, 'testing', OUTPUT_PATH, ['JK01', 'JK02', 'JK03', 'JK04'])
    # plotter.align_pupil_by_session(filter=True)
    # plotter.plot_pitch_dependency(save_figure = False, show_means_as_points = False)
        
    r'''for a in range(1,5):
        plotter = PupilPlotter(pupil_df, harp_filtered, STAGE, 'testing', OUTPUT_PATH, [f'JK0{a}'])
        plotter.set_early_sessions()
        plotter.align_pupil_by_session(filter=True)
        plotter.plot_overall_baseline_sub_aligned_pupil(show_plot = False)
        plotter.plot_baseline_sub_training(show_plot=False)
        plotter.plot_baseline_sub_testing(show_plot=False)
    
    for a in range(1,5):
        plotter = PupilPlotter(pupil_df, harp_filtered, STAGE, 'testing', OUTPUT_PATH, [f'JK0{a}'])
        plotter.set_late_sessions()
        plotter.align_pupil_by_session(filter=True)
        plotter.plot_overall_baseline_sub_aligned_pupil(show_plot = False)
        plotter.plot_baseline_sub_training(show_plot=False)
        plotter.plot_baseline_sub_testing(show_plot=False)'''
    
    

    r'''stage4 = PupilPlotter(pupil_df, harp_df, STAGE, 'testing', OUTPUT_PATH, ['JK01', 'JK02', 'JK03', 'JK04'])
    
    stage4.align_pupil_by_session()
    stage4.plot_baseline_sub_aligned_pupil_by_session(show_plot = False)
    stage4.plot_sessionwide_pupil_dilation(show_plot = False)'''
    
    r'''stage4_JK04 = PupilPlotter(pupil_df, harp_df, STAGE, 'testing', OUTPUT_PATH, ['JK04'])
    stage4_JK04.align_pupil_by_session()
    stage4_JK04.plot_overall_baseline_sub_aligned_pupil(show_plot = False)
    for a in range(1,5):
        plotter = PupilPlotter(pupil_df, harp_df, STAGE, 'testing', OUTPUT_PATH, [f'JK0{a}'])
        plotter.align_pupil_by_session()
        plotter.plot_overall_baseline_sub_aligned_pupil(show_plot = False, use_median=True)'''
    

    # # TODO Make a way to baseline the data so it starts from 0, but then only output data from 0s to 4s. This will work better with the PRET model. 
    # for a in range(1,5):
    #     plotter = PupilPlotter(pupil_df, harp_filtered, STAGE, 'testing', OUTPUT_PATH, [f'JK0{a}'])
    #     plotter.align_pupil_by_session()
    #     plotter.plot_overall_baseline_sub_aligned_pupil(show_plot = False)
    #     agg_plotter = plotter.aggregate_total(baseline_data = True)
    #     for stimulus in agg_plotter.keys():
    #         pupils_for_stimulus = agg_plotter[stimulus].dropna(axis = 0,thresh=10)
    #         pupils_for_stimulus.to_csv(fr'C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\PRET\PRET\Data\Stage{STAGE}\JK0{a}_Stage{STAGE}_filtered_{stimulus}.csv')
            