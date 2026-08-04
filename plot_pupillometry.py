import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import permutations

import behaviour as tfb
import data_io as tfio
import pupillometry as tfp
from pupillometry import PupilPlotter

SESSION_PATH = Path(r"X:\Dammy\Xdetection_mouse_hf_test\session_topology_transitive_inference_cleaned.csv")
HOME_PATH = Path(r"C:\bonsai\data\JungWoo")
OUTPUT_PATH = Path(r"C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\Outputs")
# PARQUET_DIR = Path(r'X:\Dammy\mouse_pupillometry\pickles\trans_inf_test_90Hz_hpass00_lpass0')
PARQUET_DIR = Path(r'X:\Dammy\mouse_pupillometry\pickles\trans_inf_bandpass_90Hz_hpass01_lpass4')
HARP_DIR = Path(r'X:\Dammy\harpbins')



if __name__ == "__main__":
    
    ############## STAGE 1 ###################
    
    # stage = 1
    
    # pupil_df = tfio.load_aggregate_pupil_df(SESSION_PATH, stage, PARQUET_DIR)
    # harp_df = tfio.load_aggregate_harp_df(SESSION_PATH, stage, HARP_DIR)
    # td_df = tfio.load_aggregate_trial_data(SESSION_PATH, HOME_PATH)
    
    # harp_filtered = tfio.filter_harp_by_successful_trials(harp_df, td_df, print_trial_lengths=False)
    
    # mean_peak_times = []
    # for a in range(1,5):
    #     plotter = PupilPlotter(pupil_df, harp_filtered, stage, 'testing', OUTPUT_PATH, [f'JK0{a}'])
    #     plotter.align_pupil_by_session(filter=True)
    #     mean_peak_time = plotter.plot_stage1_peak_time_by_stimulus(search_window= (0.15, 0.75), show_plot= False)
    #     mean_peak_times.append(mean_peak_time)
    # print(f"Overall mean peak time across animals: {np.mean(mean_peak_times):.3f} s")
    
    
    # for a in range(1,5):
    #     plotter = PupilPlotter(pupil_df, harp_filtered, stage, 'testing', OUTPUT_PATH, [f'JK0{a}'])
    #     plotter.align_pupil_by_session(filter=True)
        # plotter.plot_distribution_by_session(show_plot=False)
        # if a == 2: 
        #     plotter.plot_stage1_window_diagram(stim_to_show = 'G', show_plot=False)
        # # plotter.plot_overall_distribution(show_plot=False)
        # plotter.plot_overall_baseline_sub_aligned_pupil(show_plot = False)
        # plotter.plot_sessionwide_pupil_dilation(show_plot=False)
        # plotter.plot_pitch_dependency(offset=0.5, window_size=0.1, show_plot=False, plot_quad=False)
    
    # plotter = PupilPlotter(pupil_df, harp_filtered, stage, 'testing', OUTPUT_PATH, ['JK01', 'JK02', 'JK03', 'JK04'])
    # plotter.align_pupil_by_session(filter=True)
    # # plotter.plot_overall_distribution(show_plot=False)
    # # plotter.plot_overall_baseline_sub_aligned_pupil(show_plot = False)
    # # plotter.plot_pitch_dependency(offset = 0.5, window_size=0.1, show_plot=False, plot_quad=False)
    # # plotter.plot_baseline_sub_aligned_pupil_by_session(show_plot=False)
    
    
    
    ################ STAGE 4 ################
        
    # stage = 4
    
    # pupil_df = tfio.load_aggregate_pupil_df(SESSION_PATH, stage, PARQUET_DIR)
    # harp_df = tfio.load_aggregate_harp_df(SESSION_PATH, stage, HARP_DIR)
    # td_df = tfio.load_aggregate_trial_data(SESSION_PATH, HOME_PATH)
    
    # harp_filtered = tfio.filter_harp_by_successful_trials(harp_df, td_df, print_trial_lengths=False)
        
    # for a in range(1,5):
    #     plotter = PupilPlotter(pupil_df, harp_filtered, stage, 'testing', OUTPUT_PATH, [f'JK0{a}'])
    #     plotter.align_pupil_by_session(filter=True)
    #     # plotter.plot_distribution_by_session(show_plot=False)
    #     # plotter.plot_overall_distribution(show_plot=False)
    #     if a == 2:
    #         plotter.plot_differences_method('CDEF', 'CFED', (0.5,3), show_plot = False)
    #     # plotter.plot_sessionwide_pupil_dilation(show_plot=False)
    #     plotter.plot_difference('CDEF', 'CFED', (0.5,3), regress_baseline=False)
        # plotter.plot_cosine_similarity('ABCD', 'EFGH', (0, 2.5))
        # plotter.plot_overall_baseline_sub_aligned_pupil(show_plot = False)
        # plotter.plot_baseline_sub_testing(show_plot = False)
        # if a == 1: 
        #     plotter.plot_raw_pupil(highpass_cutoff=0.1, lowpass_cutoff = 4)
        #     plotter.plot_pupil_preprocessing(session_id = '260603_000', show_plot = False)
        # plotter.plot_overall_baseline_regressed_pupil(show_plot= False)
    
    # plotter = PupilPlotter(pupil_df, harp_filtered, stage, 'testing', OUTPUT_PATH, ['JK01', 'JK02', 'JK03', 'JK04'])
    # plotter.align_pupil_by_session(filter=True)
    # # plotter.plot_overall_distribution(show_plot=False)
    # # plotter.plot_overall_baseline_sub_aligned_pupil(show_plot = False)
    # # plotter.plot_overall_baseline_regressed_pupil(show_plot= False)
    # # plotter.plot_baseline_sub_aligned_pupil_by_session(show_plot=False)
    
    
    
    ############## STAGE 5 ###################
        
    stage = 5
    
    pupil_df = tfio.load_aggregate_pupil_df(SESSION_PATH, stage, PARQUET_DIR)
    harp_df = tfio.load_aggregate_harp_df(SESSION_PATH, stage, HARP_DIR)
    td_df = tfio.load_aggregate_trial_data(SESSION_PATH, HOME_PATH)
    
    harp_filtered = tfio.filter_harp_by_successful_trials(harp_df, td_df, print_trial_lengths=False)
    
    plotter = PupilPlotter(pupil_df, harp_filtered, stage, 'testing', OUTPUT_PATH, ['JK02'])
    plotter.plot_pupil_preprocessing('260625_000', show_plot=False)
    
    # for a in range(1,5):
    #     plotter = PupilPlotter(pupil_df, harp_filtered, stage, 'testing', OUTPUT_PATH, [f'JK0{a}'])
    #     plotter.align_pupil_by_session(filter=True)
    #     # plotter.plot_distribution_by_session(show_plot=False)
    #     # plotter.plot_overall_distribution(show_plot=False)
    #     # plotter.plot_sessionwide_pupil_dilation(show_plot=False)
    #     # plotter.plot_overall_baseline_sub_aligned_pupil(show_plot = False)
    #     # plotter.plot_overall_baseline_regressed_pupil(show_plot=False)
    #     plotter.plot_stage5_perms(show_plot=False)
        
    # plotter = PupilPlotter(pupil_df, harp_filtered, stage, 'testing', OUTPUT_PATH, ['JK01', 'JK02', 'JK03', 'JK04'])
    # plotter.align_pupil_by_session(filter=True)
    # # plotter.plot_overall_distribution(show_plot=False)
    # # plotter.plot_overall_baseline_sub_aligned_pupil(show_plot = False)
    # # plotter.plot_overall_baseline_regressed_pupil(show_plot=False)
    # plotter.plot_stage5_perms(show_plot=False)
    
    
    
    ################ GENERATE DATA FOR PRET MODEL #############################
    # # TODO Make a way to baseline the data so it starts from 0, but then only output data from 0s to 4s. This will work better with the PRET model. 
    # for a in range(1,5):
    #     plotter = PupilPlotter(pupil_df, harp_filtered, STAGE, 'testing', OUTPUT_PATH, [f'JK0{a}'])
    #     plotter.align_pupil_by_session()
    #     plotter.plot_overall_baseline_sub_aligned_pupil(show_plot = False)
    #     agg_plotter = plotter.aggregate_total(baseline_data = True)
    #     for stimulus in agg_plotter.keys():
    #         pupils_for_stimulus = agg_plotter[stimulus].dropna(axis = 0,thresh=10)
    #         pupils_for_stimulus.to_csv(fr'C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\PRET\PRET\Data\Stage{STAGE}\JK0{a}_Stage{STAGE}_filtered_{stimulus}.csv')
    
    
