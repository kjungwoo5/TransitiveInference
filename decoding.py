import numpy as np

from pathlib import Path
import sys
sys.path.append('../')
sys.path.append(str(Path(__file__).parent.parent.parent))
from Analysis.XdetectionCore.xdetectioncore.decoding.decoding_funcs import Decoder

import matplotlib as mpl
import matplotlib.pyplot as plt
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
    
    plotter = PupilPlotter(pupil_df, harp_filtered, STAGE, 'testing', OUTPUT_PATH, [f'JK03'])
    plotter.align_pupil_by_session(filter=True)
    
    # [0.04: 0.0600438596491228, 0.1: 0.06302631578947368, 0.2: 0.06298245614035088, 0.5: 0.06109649122807017]
    # [0.02: 0.06114035087719298, 0.7: 0.05837719298245615, 0.9: 0.05530701754385965]
    # [1: 0.04956140350877193, 1.5: 0.05263157894736842]
    for window_size in [0.1]:
        print(f'window size: {window_size}')
        pip_df = plotter.prep_for_decoding(window_size=window_size)
        
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
        
        # print(subsampled_first['first_tone'].value_counts())
        # print(subsampled_second['second_tone'].value_counts())
        # print(subsampled_third['third_tone'].value_counts())
        # print(subsampled_fourth['fourth_tone'].value_counts())
        # print(subsampled_stimuli['stimulus_id'].value_counts())
        
        
        print('Decoding for first tone position: ')
        predictors = subsampled_first[['pip1', 'pip2', 'pip3', 'pip4']].to_numpy(dtype=float)
        features = subsampled_first['first_tone'].to_numpy()
        
        decoder = Decoder(predictors=predictors, features=features, model_name="svc")
        
        decoder.decode(
        dec_kwargs={
            "cv_folds": 5,      # 5-fold cross-validation
            "n_runs": 10       # repeat a few times
            }
        )

        # Inspect results
        print("Mean accuracy:", np.mean(decoder.accuracy))
        decoder.plot_confusion_matrix(labels = subsampled_first['first_tone'].unique())
        plt.show()
        # print("Fold accuracies:", decoder.fold_accuracy)
        
        
        print('Decoding for second tone position: ')
        predictors = subsampled_second[['pip1', 'pip2', 'pip3', 'pip4']].to_numpy(dtype=float)
        features = subsampled_second['second_tone'].to_numpy()
        
        decoder = Decoder(predictors=predictors, features=features, model_name="svc")
        
        decoder.decode(
        dec_kwargs={
            "cv_folds": 5,      # 5-fold cross-validation
            "n_runs": 10       # repeat a few times
            }
        )

        # Inspect results
        print("Mean accuracy:", np.mean(decoder.accuracy))
        
        print('Decoding for third tone position: ')
        predictors = subsampled_third[['pip1', 'pip2', 'pip3', 'pip4']].to_numpy(dtype=float)
        features = subsampled_third['third_tone'].to_numpy()
        
        decoder = Decoder(predictors=predictors, features=features, model_name="svc")
        
        decoder.decode(
        dec_kwargs={
            "cv_folds": 5,      # 5-fold cross-validation
            "n_runs": 10       # repeat a few times
            }
        )

        # Inspect results
        print("Mean accuracy:", np.mean(decoder.accuracy))
        
        print('Decoding for fourth tone position: ')
        predictors = subsampled_fourth[['pip1', 'pip2', 'pip3', 'pip4']].to_numpy(dtype=float)
        features = subsampled_fourth['fourth_tone'].to_numpy()
        
        decoder = Decoder(predictors=predictors, features=features, model_name="svc")
        
        decoder.decode(
        dec_kwargs={
            "cv_folds": 5,      # 5-fold cross-validation
            "n_runs": 10       # repeat a few times
            }
        )

        # Inspect results
        print("Mean accuracy:", np.mean(decoder.accuracy))
        
        print('Decoding for sequence identity: ')
        predictors = subsampled_stimuli[['pip1', 'pip2', 'pip3', 'pip4']].to_numpy(dtype=float)
        features = subsampled_stimuli['stimulus_id'].to_numpy()
        
        decoder = Decoder(predictors=predictors, features=features, model_name="svc")
        
        decoder.decode(
        dec_kwargs={
            "cv_folds": 5,      # 5-fold cross-validation
            "n_runs": 10       # repeat a few times
            }
        )

        # Inspect results
        print("Mean accuracy:", np.mean(decoder.accuracy))
        decoder.plot_confusion_matrix(labels = subsampled_stimuli['stimulus_id'].unique())
        plt.show()

quit()

# Sample output from Claude!

# A few important notes:

# The input matrix should be shape (n_samples, n_features).
# If you have a single pupillometry time series, you should first derive features from it, such as:
# mean pupil size
# peak dilation
# latency to peak
# slope
# area under the curve

# Example: 100 samples, 5 pupillometry features
# Each row = one sample/trial/window
# Each column = one feature (e.g. mean pupil size, peak dilation, latency, slope, etc.)
X = np.random.randn(100, 5)

# Labels for each sample
y = np.array([0, 1] * 50)

# Create decoder
decoder = Decoder(
    predictors=X,
    features=y,
    model_name="svc"
)

# Run decoding
decoder.decode(
    dec_kwargs={
        "cv_folds": 5,      # 5-fold cross-validation
        "n_runs": 10       # repeat a few times
    }
)

# Inspect results
print("Mean accuracy:", np.mean(decoder.accuracy))
print("Fold accuracies:", decoder.fold_accuracy)




# Example data structure:
# 80 trials, 8 timepoints per trial
pupil_features = np.array([
    [0.42, 0.45, 0.51, 0.58, 0.62, 0.59, 0.54, 0.50],  # trial 1
    [0.38, 0.41, 0.47, 0.55, 0.61, 0.57, 0.52, 0.48],  # trial 2
    ...
])

labels = np.array([0, 1, 0, 1, 0, 1, ...])

# Reshape to samples x features
# Here each trial is one sample, and each timepoint is one feature
X = pupil_features
y = labels

decoder = Decoder(predictors=X, features=y, model_name="svc")
decoder.decode(dec_kwargs={"cv_folds": 5, "n_runs": 20})