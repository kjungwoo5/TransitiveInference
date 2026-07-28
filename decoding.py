import numpy as np

import json

from pathlib import Path
import sys
sys.path.append('../')
sys.path.append(str(Path(__file__).parent.parent.parent))
from Analysis.XdetectionCore.xdetectioncore.decoding.decoding_funcs import Decoder

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


def plot_decoder_pca_classifications(predictors, true_labels, predicted_labels, output_path, title):
    """Project the input predictors into 2 PCs and plot the fitted classifier's decision boundaries."""
    predictors = np.asarray(predictors, dtype=float)
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)

    if predictors.shape[0] != true_labels.shape[0] or predictors.shape[0] != predicted_labels.shape[0]:
        raise ValueError("Predictors and labels must have the same number of samples.")

    pca = PCA(n_components=2)
    projected = pca.fit_transform(predictors)

    classes = np.unique(predicted_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(classes), 1)))

    fig, ax = plt.subplots(figsize=(8, 6))

    if len(classes) >= 2:
        classifier = svm.SVC(kernel='linear', probability=False, decision_function_shape='ovr')
        classifier.fit(projected, predicted_labels)

        x_min, x_max = projected[:, 0].min() - 0.5, projected[:, 0].max() + 0.5
        y_min, y_max = projected[:, 1].min() - 0.5, projected[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
        zz = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])

        if zz.ndim == 1:
            zz = zz.reshape(-1, 1)

        for class_idx in range(zz.shape[1]):
            contour = ax.contour(
                xx, yy, zz[:, class_idx].reshape(xx.shape),
                levels=[0],
                colors=colors[class_idx % len(colors)],
                linewidths=1.8,
                alpha=0.9,
            )
            ax.clabel(contour, inline=True, fontsize=8, fmt=lambda _: '')

    for class_idx, class_name in enumerate(classes):
        mask = predicted_labels == class_name
        ax.scatter(
            projected[mask, 0],
            projected[mask, 1],
            s=60,
            color=colors[class_idx % len(colors)],
            alpha=0.85,
            edgecolor='k',
            linewidth=0.3,
            label=str(class_name),
        )

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(title)
    ax.legend(title='Predicted class', bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    
    STAGE = 5
    
    pupil_df = tfio.load_aggregate_pupil_df(SESSION_PATH, STAGE, PARQUET_DIR)
    harp_df = tfio.load_aggregate_harp_df(SESSION_PATH, STAGE, HARP_DIR)
    td_df = tfio.load_aggregate_trial_data(SESSION_PATH, HOME_PATH)
    
    
    harp_filtered = tfio.filter_harp_by_successful_trials(harp_df, td_df, print_trial_lengths=False)
    
    
    for animal in ['JK01', 'JK02', 'JK03', 'JK04']:
        plotter = PupilPlotter(pupil_df, harp_filtered, STAGE, 'testing', OUTPUT_PATH, [animal])
        plotter.align_pupil_by_session(filter=True)
        
        
        window_sizes = [0, 0.5] # np.linspace(0, 1, 21)
        offsets = [0.25, 0.38, 0.5, 0.63, 0.75]
        
        for offset in offsets: 
            accuracies_by_window = {}
        
            for window_size in window_sizes:
                print(f'\nwindow size: {window_size}')
                pip_df = plotter.prep_for_decoding(window_size=window_size, tmax= 0.5)
                
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
                print(subsampled_stimuli['stimulus_id'].value_counts())
                
                
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
                accuracies_by_window[window_size] = [np.mean(decoder.accuracy)]
                
                decoder.plot_confusion_matrix(labels = subsampled_first['first_tone'].unique())
                plt.savefig(OUTPUT_PATH / "Decoding" / f"Stage5_{animal}_first_tone_position_confusion_matrix_{window_size}.png")

                
                
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
                decoder.plot_confusion_matrix(labels = subsampled_second['second_tone'].unique())
                plt.savefig(OUTPUT_PATH / "Decoding" / f"Stage5_{animal}_second_tone_confusion_matrix_{window_size}.png")
                accuracies_by_window[window_size].append(np.mean(decoder.accuracy))

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
                decoder.plot_confusion_matrix(labels = subsampled_third['third_tone'].unique())
                plt.savefig(OUTPUT_PATH / "Decoding" / f"Stage5_{animal}_third_tone_confusion_matrix_{window_size}.png")
                accuracies_by_window[window_size].append(np.mean(decoder.accuracy))
                
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
                decoder.plot_confusion_matrix(labels = subsampled_fourth['fourth_tone'].unique())
                plt.savefig(OUTPUT_PATH / "Decoding" / f"Stage5_{animal}_fourth_tone_confusion_matrix_{window_size}.png")
                accuracies_by_window[window_size].append(np.mean(decoder.accuracy))
                
                
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
                plt.savefig(OUTPUT_PATH / "Decoding" / f"Stage5_{animal}_sequence_identity_confusion_matrix_{window_size}.png")

                
                accuracies_by_window[window_size].append(np.mean(decoder.accuracy))

            print(accuracies_by_window)
            with open(f'accuracies_by_window_{animal}_{offset}.json', 'w') as f:
                json.dump(accuracies_by_window, f)