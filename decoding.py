import os
import numpy as np

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

SESSION_PATH = Path(r"X:\Dammy\Xdetection_mouse_hf_test\session_topology_transitive_inference_cleaned.csv")
HOME_PATH = Path(r"C:\bonsai\data\JungWoo")
OUTPUT_PATH = Path(r"C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\Outputs")
# PARQUET_DIR = Path(r'X:\Dammy\mouse_pupillometry\pickles\trans_inf_test_90Hz_hpass00_lpass0')
PARQUET_DIR = Path(r'X:\Dammy\mouse_pupillometry\pickles\trans_inf_bandpass_90Hz_hpass01_lpass4')
HARP_DIR = Path(r'X:\Dammy\harpbins')


def _run_decoder_task(label, predictors, features, animal, window_size, output_path, labels, save_suffix, save_fig=False):
    """Run one decoder fit without touching Matplotlib in worker threads."""
    decoder = Decoder(predictors=predictors, features=features, model_name="svc")
    decoder.decode(
        dec_kwargs={"cv_folds": 5, "n_runs": 10},
        parallel_flag=False,
    )

    accuracy = float(np.mean(decoder.accuracy))
    return label, accuracy, decoder


if __name__ == "__main__":
    
    STAGE = 5
    
    pupil_df = tfio.load_aggregate_pupil_df(SESSION_PATH, STAGE, PARQUET_DIR)
    harp_df = tfio.load_aggregate_harp_df(SESSION_PATH, STAGE, HARP_DIR)
    td_df = tfio.load_aggregate_trial_data(SESSION_PATH, HOME_PATH)
    
    
    harp_filtered = tfio.filter_harp_by_successful_trials(harp_df, td_df, print_trial_lengths=False)
    
    
    for animal in ['JK01', 'JK02', 'JK03', 'JK04']:
        plotter = PupilPlotter(pupil_df, harp_filtered, STAGE, 'testing', OUTPUT_PATH, [animal])
        plotter.align_pupil_by_session(filter=True)
        
        window_sizes = np.linspace(0.05, 0.5, 10)
        accuracies_by_window = {}
        
        for window_size in window_sizes:
            print(f'\nwindow size: {window_size}')
            pip_df = plotter.prep_for_decoding(tmax= 0.49, window_size=window_size, time_from_next_pip=None)
            pip_df.dropna(inplace=True)
            
            pip_df = pip_df[pip_df['stimulus_id'] != 'X']
            pip_df = pip_df[pip_df['stimulus_id'] != 'GHAB']
            pip_df = pip_df[pip_df['stimulus_id'] != 'ABGH']
            
            pip_df['first_tone'] = pip_df['stimulus_id'].str[0]
            pip_df['second_tone'] = pip_df['stimulus_id'].str[1]
            pip_df['third_tone'] = pip_df['stimulus_id'].str[2]
            pip_df['fourth_tone'] = pip_df['stimulus_id'].str[3]
            pip_df['remaining_sequence'] = pip_df['stimulus_id'].str[1:4]
            
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
            
            subsampled_rems = {}
            # Sample by min number of remaining sequences per starting tone
            for index, letter in enumerate('CDEF'):
                min_remaining_seq = pip_df[pip_df['first_tone'] == letter]['remaining_sequence'].value_counts().min()
                subsampled_rems[letter] = pip_df[pip_df['first_tone'] == letter].groupby('remaining_sequence').sample(min_remaining_seq)
                subsampled_rems[letter].reset_index(inplace=True)
                
                
            print(subsampled_first['first_tone'].value_counts())
            print(subsampled_second['second_tone'].value_counts())
            print(subsampled_third['third_tone'].value_counts())
            print(subsampled_fourth['fourth_tone'].value_counts())
            print(subsampled_stimuli['stimulus_id'].value_counts())
            for letter in 'CDEF':
                print(subsampled_rems[letter]['remaining_sequence'].value_counts())
            
            
            task_specs = [
                (
                    'first_tone_position',
                    subsampled_first[['pip1']].to_numpy(dtype=float),
                    subsampled_first['first_tone'].to_numpy(),
                    animal,
                    window_size,
                    OUTPUT_PATH,
                    subsampled_first['first_tone'].unique(),
                    'first_tone_position',
                ),
                (
                    'second_tone_position',
                    subsampled_second[['pip1', 'pip2']].to_numpy(dtype=float),
                    subsampled_second['second_tone'].to_numpy(),
                    animal,
                    window_size,
                    OUTPUT_PATH,
                    subsampled_second['second_tone'].unique(),
                    'second_tone_position',
                ),
                (
                    'third_tone_position',
                    subsampled_third[['pip1', 'pip2', 'pip3']].to_numpy(dtype=float),
                    subsampled_third['third_tone'].to_numpy(),
                    animal,
                    window_size,
                    OUTPUT_PATH,
                    subsampled_third['third_tone'].unique(),
                    'third_tone_position',
                ),
                (
                    'fourth_tone_position',
                    subsampled_fourth[['pip1', 'pip2', 'pip3', 'pip4']].to_numpy(dtype=float),
                    subsampled_fourth['fourth_tone'].to_numpy(),
                    animal,
                    window_size,
                    OUTPUT_PATH,
                    subsampled_fourth['fourth_tone'].unique(),
                    'fourth_tone_position',
                ),
                (
                    'sequence_identity',
                    subsampled_stimuli[['pip1', 'pip2', 'pip3', 'pip4']].to_numpy(dtype=float),
                    subsampled_stimuli['stimulus_id'].to_numpy(),
                    animal,
                    window_size,
                    OUTPUT_PATH,
                    subsampled_stimuli['stimulus_id'].unique(),
                    'sequence_identity',
                ),
                (
                    'C_remaining_sequence',
                    subsampled_rems['C'][['pip1', 'pip2', 'pip3', 'pip4']].to_numpy(dtype=float),
                    subsampled_rems['C']['remaining_sequence'].to_numpy(),
                    animal,
                    window_size,
                    OUTPUT_PATH,
                    subsampled_rems['C']['remaining_sequence'].unique(),
                    'C_remaining_sequence',
                ),
                (
                    'D_remaining_sequence',
                    subsampled_rems['D'][['pip1', 'pip2', 'pip3', 'pip4']].to_numpy(dtype=float),
                    subsampled_rems['D']['remaining_sequence'].to_numpy(),
                    animal,
                    window_size,
                    OUTPUT_PATH,
                    subsampled_rems['D']['remaining_sequence'].unique(),
                    'D_remaining_sequence',
                ),
                (
                    'E_remaining_sequence',
                    subsampled_rems['E'][['pip1', 'pip2', 'pip3', 'pip4']].to_numpy(dtype=float),
                    subsampled_rems['E']['remaining_sequence'].to_numpy(),
                    animal,
                    window_size,
                    OUTPUT_PATH,
                    subsampled_rems['E']['remaining_sequence'].unique(),
                    'E_remaining_sequence',
                ),
                (
                    'F_remaining_sequence',
                    subsampled_rems['F'][['pip1', 'pip2', 'pip3', 'pip4']].to_numpy(dtype=float),
                    subsampled_rems['F']['remaining_sequence'].to_numpy(),
                    animal,
                    window_size,
                    OUTPUT_PATH,
                    subsampled_rems['F']['remaining_sequence'].unique(),
                    'F_remaining_sequence',
                ),
            ]

            n_workers = min(len(task_specs), max(1, os.cpu_count() - 1))
            results = Parallel(n_jobs=n_workers, prefer='threads')(
                delayed(_run_decoder_task)(
                    label,
                    predictors,
                    features,
                    animal,
                    window_size,
                    output_path,
                    labels,
                    save_suffix,
                )
                for label, predictors, features, animal, window_size, output_path, labels, save_suffix in task_specs
            )

            accuracies_by_window[window_size] = [accuracy for _, accuracy, _ in results]
            for task_label, task_accuracy, decoder in results:
                print(f"{task_label} mean accuracy: {task_accuracy:.4f}")

                task_info = next(
                    spec for spec in task_specs if spec[0] == task_label
                )
                task_labels = task_info[6]
                task_save_suffix = task_info[7]

                decoder.plot_confusion_matrix(labels=task_labels)
                save_path = OUTPUT_PATH / "Decoding" / f"{animal}_{task_save_suffix}_confusion_matrix_{window_size}.svg"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path)
                plt.close("all")

        print(accuracies_by_window)
        with open(f'bandpassed_accuracies_by_window_{animal}.json', 'w') as f:
            json.dump(accuracies_by_window, f)