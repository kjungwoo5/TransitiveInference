import numpy as np
import os
import json

from pathlib import Path
import sys
sys.path.append('../')
sys.path.append(str(Path(__file__).parent.parent.parent))
from Analysis.XdetectionCore.xdetectioncore.decoding.decoding_funcs import Decoder
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from itertools import permutations
from tqdm import tqdm

import behaviour as tfb
import data_io as tfio
import pupillometry as tfp
from pupillometry import PupilPlotter

SESSION_PATH = Path(r"X:\Dammy\Xdetection_mouse_hf_test\session_topology_transitive_inference_full.csv")
HOME_PATH = Path(r"C:\bonsai\data\JungWoo")
OUTPUT_PATH = Path(r"C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\Outputs")
PARQUET_DIR = Path(r'X:\Dammy\mouse_pupillometry\pickles\trans_inf_test_90Hz_hpass00_lpass0')
HARP_DIR = Path(r'X:\Dammy\harpbins')


def plot_permutation_results(results, title, output_path):
    """Plot a histogram of permuted accuracies and mark the observed accuracy."""
    plt.figure(figsize=(7, 4.5))
    plt.hist(results["permuted_accuracies"], bins=20, color="steelblue", edgecolor="black", alpha=0.8)
    plt.axvline(results["observed_accuracy"], color="red", linestyle="--", linewidth=2, label="Observed accuracy")
    plt.axvline(results["permutation_mean"], color="black", linestyle=":", linewidth=1.5, label="Permutation mean")
    plt.xlabel("Decoder accuracy")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()

    p_value = results["p_value"]
    plt.text(
        0.98,
        0.95,
        f"p = {p_value:.2f}",
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_observed_vs_permutation_boxplot(results_by_animal, output_path, data_type = "first_tone"):
    """Plot paired observed and permutation-mean accuracies for each animal as a boxplot."""
    animals = []
    observed_values = []
    permutation_mean_values = []
    
    plot_type = {"first_tone": "first tone",
                 "sequence_identity": "sequence identity"}

    for animal, animal_results in results_by_animal.items():
        data_results = animal_results.get(data_type, {})
        if not data_results:
            continue

        animals.append(animal)
        observed_values.append(data_results.get("observed_accuracy", np.nan))
        permutation_mean_values.append(data_results.get("permutation_mean", np.nan))

    if not animals:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = [observed_values, permutation_mean_values]

    box = ax.boxplot(data, labels=["Observed accuracy", "Permutation mean"], patch_artist=True)
    for patch in box["boxes"]:
        patch.set(facecolor="lightgray", alpha=0.8)

    for idx, animal in enumerate(animals):
        ax.plot([1, 2], [observed_values[idx], permutation_mean_values[idx]], color="tab:red", alpha=0.7, linewidth=1.5)
        ax.scatter(1, observed_values[idx], color="tab:blue", s=50, zorder=3)
        ax.scatter(2, permutation_mean_values[idx], color="tab:orange", s=50, zorder=3)
        # ax.text(1.5, np.mean([observed_values[idx], permutation_mean_values[idx]]), animal,
        #         ha="center", va="bottom", fontsize=8, color="black")

    ax.set_ylabel("Decoder accuracy")
    ax.set_title(f"Observed vs permutation-mean accuracy of {plot_type.get(data_type, data_type)} by animal")
    # ax.set_ylim(0, 1.05)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def _compute_permuted_accuracy(predictors, features, cv_folds, n_runs, seed):
    rng = np.random.default_rng(seed)
    permuted_features = rng.permutation(features)
    perm_decoder = Decoder(predictors=predictors, features=permuted_features, model_name="svc")
    perm_decoder.decode(dec_kwargs={"cv_folds": cv_folds, "n_runs": n_runs})
    return float(np.mean(perm_decoder.accuracy))


def run_permutation_test(predictors, features, n_permutations=99, cv_folds=5, n_runs=10, n_jobs=None):
    """Run a label-permutation decoder test by retraining the decoder for each shuffled label set."""
    observed_decoder = Decoder(predictors=predictors, features=features, model_name="svc")
    observed_decoder.decode(dec_kwargs={"cv_folds": cv_folds, "n_runs": n_runs})
    observed_accuracy = float(np.mean(observed_decoder.accuracy))

    if n_jobs is None:
        n_jobs = max(1, min(n_permutations, (os.cpu_count() or 2) - 1))

    rng = np.random.default_rng(0)
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_permutations, dtype=np.int32)

    with tqdm_joblib(tqdm(desc="Permutation tests", total=n_permutations, unit="perm")):
        permuted_accuracies = Parallel(n_jobs=n_jobs)(
            delayed(_compute_permuted_accuracy)(predictors, features, cv_folds, n_runs, int(seed))
            for seed in seeds
        )

    permuted_accuracies = np.asarray(permuted_accuracies)
    p_value = (np.sum(permuted_accuracies >= observed_accuracy) + 1) / (n_permutations + 1)

    return {
        "observed_accuracy": observed_accuracy,
        "permuted_accuracies": permuted_accuracies,
        "permutation_mean": float(np.mean(permuted_accuracies)),
        "permutation_std": float(np.std(permuted_accuracies)),
        "p_value": p_value,
    }


if __name__ == "__main__":
    
    STAGE = 5
    window_size = 0.5
    
    pupil_df = tfio.load_aggregate_pupil_df(SESSION_PATH, STAGE, PARQUET_DIR)
    harp_df = tfio.load_aggregate_harp_df(SESSION_PATH, STAGE, HARP_DIR)
    td_df = tfio.load_aggregate_trial_data(SESSION_PATH, HOME_PATH)
    
    harp_filtered = tfio.filter_harp_by_successful_trials(harp_df, td_df, print_trial_lengths=False)
    
    # ~1hr per permutation test for 100 permutations, 5-fold CV, and 10 runs.
    # With parallelisation, ~10 minutes per 100 perms. 
    
    results_by_animal = {}
    
    for animal in ['JK01', 'JK02', 'JK03', 'JK04']:
        plotter = PupilPlotter(pupil_df, harp_filtered, STAGE, 'testing', OUTPUT_PATH, [animal])
        plotter.align_pupil_by_session(filter=True)
        
        
        accuracies_by_window = {}
        
        print(f'\nAnimal: {animal}')
        pip_df = plotter.prep_for_decoding(window_size=0.5)
        
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
        
        
        results_by_animal[animal] = {}

        print('Decoding for first tone position: ')
        predictors = subsampled_first[['pip1', 'pip2', 'pip3', 'pip4']].to_numpy(dtype=float)
        features = subsampled_first['first_tone'].to_numpy()

        first_tone_results = run_permutation_test(
            predictors=predictors,
            features=features,
            n_permutations=99,
            cv_folds=5,
            n_runs=10,
        )

        print("Observed accuracy:", first_tone_results["observed_accuracy"])
        print("Permutation mean accuracy:", first_tone_results["permutation_mean"])
        print("Permutation std:", first_tone_results["permutation_std"])
        print("Permutation p-value:", first_tone_results["p_value"])
        plot_permutation_results(
            first_tone_results,
            title=f"{animal} - First tone permutation test",
            output_path=OUTPUT_PATH / "Permutation Tests" / f"{animal}_first_tone_permutation_hist.png",
        )
        results_by_animal[animal]["first_tone"] = {
            "observed_accuracy": first_tone_results["observed_accuracy"],
            "permutation_mean": first_tone_results["permutation_mean"],
            "permutation_std": first_tone_results["permutation_std"],
            "p_value": first_tone_results["p_value"],
        }
        accuracies_by_window[window_size] = [first_tone_results["observed_accuracy"]]
        
        # decoder.plot_confusion_matrix(labels = subsampled_first['first_tone'].unique())
        # plt.show()
        # print("Fold accuracies:", decoder.fold_accuracy)
        
        
        print('Decoding for sequence identity: ')
        predictors = subsampled_stimuli[['pip1', 'pip2', 'pip3', 'pip4']].to_numpy(dtype=float)
        features = subsampled_stimuli['stimulus_id'].to_numpy()

        sequence_results = run_permutation_test(
            predictors=predictors,
            features=features,
            n_permutations=99,
            cv_folds=5,
            n_runs=10,
        )

        print("Observed accuracy:", sequence_results["observed_accuracy"])
        print("Permutation mean accuracy:", sequence_results["permutation_mean"])
        print("Permutation std:", sequence_results["permutation_std"])
        print("Permutation p-value:", sequence_results["p_value"])
        plot_permutation_results(
            sequence_results,
            title=f"{animal} - Sequence identity permutation test",
            output_path=OUTPUT_PATH / "Permutation Tests" / f"{animal}_sequence_identity_permutation_hist.png",
        )
        results_by_animal[animal]["sequence_identity"] = {
            "observed_accuracy": sequence_results["observed_accuracy"],
            "permutation_mean": sequence_results["permutation_mean"],
            "permutation_std": sequence_results["permutation_std"],
            "p_value": sequence_results["p_value"],
        }
        # decoder.plot_confusion_matrix(labels = subsampled_stimuli['stimulus_id'].unique())
        # plt.show()
    

    plot_observed_vs_permutation_boxplot(
        results_by_animal,
        OUTPUT_PATH / "Permutation Tests" / "sequence_identity_observed_vs_permutation_mean_boxplot.png",
        data_type = 'sequence_identity'
    )

    plot_observed_vs_permutation_boxplot(
        results_by_animal,
        OUTPUT_PATH / "Permutation Tests" / "first_tone_observed_vs_permutation_mean_boxplot.png",
        data_type = 'first_tone'
    )
    
    # Save results to a JSON file
    with open(OUTPUT_PATH / "Permutation Tests" / "permutation_test_results.json", "w") as f:
        json.dump(results_by_animal, f, indent=4)
