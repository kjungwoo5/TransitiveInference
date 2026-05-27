import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import scipy
import time
import pickle

from tqdm import tqdm
from copy import copy

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from xdetectioncore.behaviour import get_main_sess_td_df
from xdetectioncore.paths import posix_from_win
from xdetectioncore.ephys.aggregate_ephys_funcs import load_aggregate_td_df
from xdetectioncore.io_utils import load_pupil_sess_lazy
from xdetectioncore.plotting import plot_shaded_error_ts, format_axis
plt.ion()


# Paths to data files
SESSION_PATH = Path(r"X:\Dammy\Xdetection_mouse_hf_test\session_topology_transitiveinference_pilot.csv")
PARQUET_DIR = Path(r'X:\Dammy\mouse_pupillometry\pickles\trans_inf_test_90Hz_hpass00_lpass0')
HARP_DIR = Path(r'X:\Dammy\harpbins')
DATA_DIR = Path(r"C:\bonsai\data\JungWoo")


# Modifiable parameters
STAGE = 3

ANIMALS_TO_DROP = ['JK02', 'JK03', 'JK04']

SESSIONS_TO_DROP = [
    'JK04_260427', 'JK04_260428', 'JK04_260429'
    ]

SECOND_PATTERN_ONLY = True

READING_WINDOW = [-2, 5]
PLOTTING_WINDOW = [-1, 4]

LOAD_FROM_CSV = True

PLOT_BY_SESSION = False

SHOW_DISTRIBUTION = True

# Defined variables based on modified parameters
STIMULI = ['X', 'ABCD', 'CDEF', 'GHIJ', 'EFGH', 'EFHG', 'BCDE']
ANIMALS = [x for x in ['JK01', 'JK02', 'JK03', 'JK04'] if x not in ANIMALS_TO_DROP]
animals_to_list = ', '.join(ANIMALS)
DARK_SESSIONS = ['JK01_260401','JK01_260402','JK01_260403','JK01_260406','JK01_260407','JK01_260408','JK01_260409',
    'JK02_260401','JK02_260402','JK02_260403','JK02_260406','JK02_260407','JK02_260408']
SESSIONS_TO_DROP.extend(DARK_SESSIONS)

STIMULUS_COLOURS = {
    'X': 'k',
    'ABCD': 'c',
    'CDEF': 'm',
    'GHIJ': 'y',
    'EFGH': 'b',
    'EFHG': 'r',
    'BCDE': 'g'
}

Y_LIMS = {
    'JK01': (-0.25,0.35),
    'JK02': (-0.5,0.7),
    'JK03': (-0.5,0.4),
    'JK04': (-0.5,0.6),
    ' JK01, JK02, JK03, JK04 ': (-0.35,0.35)
}


# Functions for loading and processing data
def load_aggregate_pupil_df(session_topology: pd.DataFrame, stage: int, parquet_dir: Path) -> pd.DataFrame:
    """Load pupil data for all sessions in session_topology matching the requested stage."""
    
    print(f'Loading pupil data for Stage {stage} from parquet directory...')

    session_topology = session_topology.dropna(how='all').reset_index(drop=True)

    relevant_sessions = session_topology[session_topology['Stage'] == stage].copy()
    if relevant_sessions.empty:
        raise ValueError(f'No sessions found for STAGE={stage}')

    relevant_sessions = relevant_sessions.dropna(subset=['sound_bin']).reset_index(drop=True)
    session_keys = [Path(str(sound_bin).replace('_SoundData', '')).stem
                    for sound_bin in relevant_sessions['sound_bin']]

    pupil_store = load_pupil_sess_lazy(parquet_dir)
    if not hasattr(pupil_store, 'keys') or len(pupil_store.keys()) == 0:
        raise FileNotFoundError(f'Parquet store not found or empty at {parquet_dir}')

    pupil_dfs = []
    missing_sessions = []
    for sess_key in session_keys:
        try:
            pupil_dfs.append(pupil_store[sess_key].pupildf)
        except KeyError:
            missing_sessions.append(sess_key)

    if missing_sessions:
        print(f'Warning: the following sessions were not found in the parquet store: {missing_sessions}')

    if not pupil_dfs:
        raise ValueError('No pupil DataFrames could be loaded for the requested stage.')

    pupil_df = pd.concat(pupil_dfs, axis=0)
    return pupil_df


def load_aggregate_harp_df(session_topology: pd.DataFrame, stage: int, harp_dir: Path) -> pd.DataFrame:
    """Load all sound_index files for sessions in the session_topology into one DataFrame."""

    print(f'Loading harp write data for Stage {stage} from harp directory...')
    
    session_topology = session_topology.dropna(how='all').reset_index(drop=True)
    relevant_sessions = session_topology[session_topology['Stage'] == stage].copy()
    if relevant_sessions.empty:
        raise ValueError(f'No sessions found for STAGE={stage}')
    
    relevant_sessions = relevant_sessions.dropna(subset=['sound_bin']).reset_index(drop=True)

    sound_bin_col = next((col for col in session_topology.columns if col.lower() == 'sound_bin'), None)
    if sound_bin_col is None:
        raise ValueError('session_topology must contain a sound_bin column')

    session_topology = session_topology.dropna(subset=[sound_bin_col]).reset_index(drop=True)
    if session_topology.empty:
        raise ValueError('No valid sound_bin values found in session_topology.')

    aggregated = []
    missing_files = []
    for _, row in relevant_sessions.iterrows():
        session_id = Path(row[sound_bin_col]).stem
        sound_index_file = harp_dir / f"{session_id}_write_indices.csv"
        if not sound_index_file.is_file():
            missing_files.append(str(sound_index_file))
            continue

        df = pd.read_csv(sound_index_file)
        df['session_id'] = session_id.replace('_SoundData', '')
        aggregated.append(df)

    if missing_files:
        print(f'Warning: missing sound_index files for {len(missing_files)} sessions:')
        for missing in missing_files:
            print(f'  {missing}')

    if not aggregated:
        raise ValueError('No sound_index files were loaded from the harp directory.')

    harp_df = pd.concat(aggregated, axis=0, ignore_index=True)
    return harp_df


def aggregate_total(aligned_pupil_by_session: dict) -> dict:
    total_responses = {}
    for stimulus in STIMULI:
        aggregate = []
        for key, value in aligned_pupil_by_session.items():
            if stimulus in aligned_pupil_by_session[key]:
                aggregate.append(aligned_pupil_by_session[key][stimulus])
        total_responses[stimulus] = pd.concat(aggregate, axis=0, ignore_index=True)
    return total_responses


def plot_by_session(aligned_pupil_by_session: dict, show_distribution: bool = False):
    for session, value in aligned_pupil_by_session.items():
        plt.pause(0.1)
        pupil_plot = plt.subplots()
        print('Plotting baseline subtracted plot for session: ', session)

        total_responses = {}
        for stimulus in STIMULI:
            aggregate = []
            for key, value in aligned_pupil_by_session[session].items():
                if stimulus == key:
                    aggregate.append(aligned_pupil_by_session[session][stimulus])
            if aggregate:
                total_responses[stimulus] = pd.concat(aggregate, axis=0, ignore_index=True)
                total_responses[stimulus] = total_responses[stimulus].tail(300)

        for event_id, response in total_responses.items():
            baseline_mean = response.loc[:, -1:0].mean(axis=1)
            baselined = response.sub(baseline_mean, axis=0)
            pupil_plot[1].plot(baselined.columns, baselined.mean(axis=0), label=event_id, color=STIMULUS_COLOURS.get(event_id, None))
            plot_shaded_error_ts(pupil_plot[1], baselined.columns, baselined.mean(axis=0),
                                 baselined.sem(axis=0), alpha=0.1, color=STIMULUS_COLOURS.get(event_id, None))
        pupil_plot[1].legend()
        pupil_plot[1].set_xlim((PLOTTING_WINDOW[0], PLOTTING_WINDOW[1]))
        annotation = f'n = {total_responses["X"].shape[0]} trials'
        pupil_plot[1].annotate(annotation, xy=(0.3, 1.02), xycoords=pupil_plot[1].get_xaxis_transform())
        pupil_plot[1].axvspan(0, 0.15, color='grey' , alpha=0.1)
        pupil_plot[1].axvspan(0.5, 0.65, color='grey', alpha=0.1)
        pupil_plot[1].axvspan(1, 1.15, color='grey', alpha=0.1)
        pupil_plot[1].axvspan(1.5, 1.65, color='grey', alpha=0.1)
        #pupil_plot[1].set_ylim((-0.5,0.9))
        pupil_plot[0].suptitle(f'Baseline subtracted plot for: {session}')
        fig = plt.gcf()
        pupil_plot[0].show()
        fig.savefig(
            fr'C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\Outputs\Second Patterns Only\Individual Sessions\Second_Pattern_{session}.png')
        
        if show_distribution:
            plt.pause(0.1)
            dist_plot = plt.subplots()
            print('Plotting distribution for session: ', session)
            actual_distribution = {}
            for stimulus in STIMULI:
                if stimulus in total_responses.keys():
                    actual_distribution[stimulus] = len(total_responses[stimulus])
                else:
                    actual_distribution[stimulus] = 0
                dist_plot[1].bar(stimulus, actual_distribution[stimulus], color=STIMULUS_COLOURS.get(stimulus, None))
                dist_plot[1].text(stimulus, actual_distribution[stimulus]+5, str(actual_distribution[stimulus]), ha='center', va='center')
            
            dist_plot[0].suptitle(f'Trial distribution for: {session}')
            annotation = f'n = {total_responses["X"].shape[0]} trials'
            dist_plot[1].annotate(annotation, xy=(0.3, 1.02), xycoords=dist_plot[1].get_xaxis_transform())
            fig = plt.gcf()
            dist_plot[0].show()
            fig.savefig(
                fr'C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\Outputs\Second Patterns Only\Actual Distributions\{session}_distribution.png'
            )
                

        

def highpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, cutoff, 'highpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data

def fit_time_resolved_baseline_regression(df:pd.DataFrame, base_window=(-1.0, 0.0)):
    """
    Fits a mass-univariate linear regression model for each time point across all trials
    to assess how baseline temporal features predict the subsequent pupil response.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame where rows are trials and columns are time points.
    base_window : tuple
        The (start, end) time for the baseline period.

    Returns:
    --------
    coef_df : pd.DataFrame
        Regression coefficients for the full model.
    total_r2 : pd.Series
        R^2 of the full model (all features combined) over time.
    feature_r2_df : pd.DataFrame
        Univariate R^2 for each feature independently over time.
    """
    # 1. Feature Engineering
    base_df = df.loc[:, base_window[0]:base_window[1]]
    t_base = base_df.columns.values

    # Feature 1: Mean
    base_mean = base_df.mean(axis=1)
    # Feature 2: Slope (Gradient)
    x = t_base - np.mean(t_base)
    base_slope = base_df.sub(base_mean, axis=0).dot(x) / np.sum(x**2)
    # Feature 3: Immediate (Last 200ms)
    imm_start = max(base_window[0], base_window[1] - 0.2)
    base_imm = df.loc[:, imm_start:base_window[1]].mean(axis=1)
    # Feature 4: Volatility (Variance)
    base_var = base_df.var(axis=1)

    X = pd.DataFrame({
        'mean': base_mean,
        'slope': base_slope,
        'immediate': base_imm,
        'volatility': base_var
    })

    # 2. Cleanup
    valid_mask = X.notna().all(axis=1) & df.notna().all(axis=1)
    X_clean = X[valid_mask]
    Y_clean = df[valid_mask]

    # 3. Full Model R2 and Coefficients
    full_model = LinearRegression()
    full_model.fit(X_clean, Y_clean)

    coef_df = pd.DataFrame(full_model.coef_, index=df.columns, columns=X_clean.columns)

    total_r2 = r2_score(Y_clean, full_model.predict(X_clean), multioutput='raw_values')
    total_r2_ser = pd.Series(total_r2, index=df.columns, name='Total_R2')

    # 4. Calculate Unique Contributions (Semi-partial R2)
    univariate_r2_results = {}

    for feature in X_clean.columns:
        # Fit a model with ONLY this one feature
        uni_model = LinearRegression().fit(X_clean[[feature]], Y_clean)

        # Calculate R2 for this single feature
        uni_r2 = r2_score(Y_clean, uni_model.predict(X_clean[[feature]]), multioutput='raw_values')
        univariate_r2_results[feature] = uni_r2

    feature_r2_df = pd.DataFrame(univariate_r2_results, index=df.columns)

    return coef_df, total_r2_ser, feature_r2_df


def regress_out_baseline(df, coef_df, base_window=(-1.0, 0.0),intercept=None):
    """
    Subtracts predicted baseline effects from pupil data using an arbitrary set of
    coefficients.

    Parameters:
    -----------
    df : pd.DataFrame
        The original trial-by-time DataFrame.
    coef_df : pd.DataFrame
        Regression coefficients (columns = feature names, index = time points).
    base_window : tuple
        The window used to calculate baseline features.
    intercept : pd.Series, optional
        The intercept (bias) of the model for each time point.

    Returns:
    --------
    regressed_df : pd.DataFrame
        The pupil data with the predicted baseline component removed.
    """
    base_df = df.loc[:, base_window[0]:base_window[1]]
    t_base = base_df.columns.values

    # Dictionary to store calculated features
    feature_map = {}

    # Dynamically calculate only the features present in coef_df
    if 'mean' in coef_df.columns:
        feature_map['mean'] = base_df.mean(axis=1)

    if 'slope' in coef_df.columns:
        x_centered = t_base - np.mean(t_base)
        feature_map['slope'] = base_df.sub(base_df.mean(axis=1), axis=0).dot(x_centered) / np.sum(x_centered**2)

    if 'immediate' in coef_df.columns:
        imm_start = max(base_window[0], base_window[1] - 0.2)
        feature_map['immediate'] = df.loc[:, imm_start:base_window[1]].mean(axis=1)

    if 'volatility' in coef_df.columns:
        feature_map['volatility'] = base_df.var(axis=1)

    if 'range' in coef_df.columns:
        feature_map['range'] = base_df.max(axis=1) - base_df.min(axis=1)

    # Construct X matrix using only the features found in coef_df
    X = pd.DataFrame({k: feature_map[k] for k in coef_df.columns if k in feature_map})

    # Calculate the prediction: (Trials x Features) dot (Features x Time)
    # Ensure columns of X match columns of coef_df exactly
    X = X[coef_df.columns]
    predicted_pupil = X.dot(coef_df.T)

    # Subtract the prediction
    regressed_df = df - predicted_pupil

    # Subtract intercept if provided (centers the data around 0)
    if intercept is not None:
        regressed_df = regressed_df.sub(intercept, axis=1)

    return regressed_df


def build_session_wide_baseline_dataframe(aligned_pupil_by_session: dict,
                                         include_events=None,
                                         smooth_window: int = 9):
    """Aggregate aligned session responses into one trial-by-time DataFrame."""
    if include_events is None:
        include_events = STIMULI

    all_responses = []
    for session, event_dict in aligned_pupil_by_session.items():
        for event_id, response in event_dict.items():
            if event_id in include_events:
                all_responses.append(response)

    if not all_responses:
        raise ValueError('No session responses found for the requested events.')

    combined = pd.concat(all_responses, axis=0, ignore_index=True)
    if smooth_window is not None and smooth_window > 1:
        combined = combined.T.rolling(window=smooth_window, min_periods=1, center=True).mean().T
    return combined


def fit_time_resolved_baseline_regression_across_sessions(aligned_pupil_by_session: dict,
                                                          base_window=(-1.0, 0.0),
                                                          include_events=None,
                                                          smooth_window: int = 9):
    """Fit baseline regression on data pooled across all sessions."""
    df_for_regr = build_session_wide_baseline_dataframe(
        aligned_pupil_by_session,
        include_events=include_events,
        smooth_window=smooth_window
    )
    return fit_time_resolved_baseline_regression(df_for_regr, base_window=base_window)


def regress_out_baseline_across_sessions(aligned_pupil_by_session: dict,
                                        coef_df: pd.DataFrame,
                                        base_window=(-1.0, 0.0),
                                        intercept=None,
                                        events_to_regress=None):
    """Apply baseline regression coefficients to each session's aligned pupil responses."""
    baselined_sessions = {}
    for session, event_dict in aligned_pupil_by_session.items():
        baselined_sessions[session] = {}
        for event_id, response in event_dict.items():
            if events_to_regress is None or event_id in events_to_regress:
                baselined_sessions[session][event_id] = regress_out_baseline(
                    copy(response), coef_df,
                    base_window=base_window,
                    intercept=intercept
                )
            else:
                baselined_sessions[session][event_id] = response.copy()
    return baselined_sessions



# Read in files depending on LOAD_FROM_CSV flag
if LOAD_FROM_CSV == False:
    pupil_df = load_aggregate_pupil_df(pd.read_csv(SESSION_PATH), STAGE, PARQUET_DIR)
    harp_df = load_aggregate_harp_df(pd.read_csv(SESSION_PATH), STAGE, HARP_DIR)
    # Write dataframes for faster loading 
    pupil_df.to_csv(fr'C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\stage{STAGE}_pupil_df.csv', index=True, header=True)
    harp_df.to_csv(fr'C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\stage{STAGE}_harp_df.csv', index=True, header=True)
elif LOAD_FROM_CSV:
    pupil_df = pd.read_csv(fr"C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\stage{STAGE}_pupil_df.csv", index_col=0)
    harp_df = pd.read_csv(fr"C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\stage{STAGE}_harp_df.csv", index_col=0)


# Filter out by sessions or animals to drop
if ANIMALS_TO_DROP:
    for animal in ANIMALS_TO_DROP:
        print(f'Dropping {animal}')
        pupil_df = pupil_df[pupil_df['session_id'].str.contains(animal) == False]
        harp_df = harp_df[harp_df['session_id'].str.contains(animal) == False]

if SESSIONS_TO_DROP:
    for session in SESSIONS_TO_DROP:
        if pupil_df['session_id'].str.contains(session).any():
            print(f'Dropping {session}')
        pupil_df = pupil_df[pupil_df['session_id'].str.contains(session) == False]
        harp_df = harp_df[harp_df['session_id'].str.contains(session) == False]





session_ids = harp_df['session_id'].unique()

aligned_pupil_by_session = {}
for session_id in session_ids:
    pupil = pupil_df[pupil_df['session_id'] == session_id]['pupilsense_raddi_a_zscored']
    harp = harp_df[harp_df['session_id'] == session_id]

    # Take pupil data from only the middle 80% of trials
    pupil = pupil.iloc[int(pupil.shape[0]*0.1):int(pupil.shape[0]*0.9)]


    # Take harp data only past the first 100 trials (i.e. occurrences of X)
    Xs = harp.index[harp['Payload'] == 3].tolist()
    if len(Xs) > 100:
        harp = harp[harp.index >= Xs[100]]
        Xs = Xs[100:]
        
    STIMULI = ['X', 'ABCD', 'CDEF', 'GHIJ', 'EFGH', 'EFHG', 'BCDE']

    if not SECOND_PATTERN_ONLY:
        ### Training stimuli ###
        # Every instance of A
        ABCDs = harp.index[harp['Payload'] == 8].tolist()

        # Every instance of C with E 2 pos forwards, and no B 1 pos back.
        CDEFs = harp.index[(harp['Payload'] == 12) & (harp['Payload'].shift(-2) == 16) & (harp['Payload'].shift(1) != 10)].tolist()

        # Every instance of G with I 2 pos forwards.
        GHIJs = harp.index[(harp['Payload'] == 20) & (harp['Payload'].shift(-2) == 24)].tolist()


        ### Testing stimuli ###
        # Every instance of E with G 2 pos forwards, and no I 4 pos forwards.
        EFGHs = harp.index[(harp['Payload'] == 16) & (harp['Payload'].shift(-2) == 20) & (harp['Payload'].shift(-4) != 24)].tolist()

        # Every instance of E with H 2 pos forwards.
        EFHGs = harp.index[(harp['Payload'] == 16) & (harp['Payload'].shift(-2) == 22)].tolist()

        # Every instance of B with no A before it.
        BCDEs = harp.index[(harp['Payload'] == 10) & (harp['Payload'].shift(1) != 8)].tolist()

    elif SECOND_PATTERN_ONLY:
        ### Training stimuli ###
        # Every instance of A not after STOP
        ABCDs = harp.index[(harp['Payload'] == 8) & (harp['Payload'].shift(1) != 30)].tolist()

        # Every instance of C not after STOP with E 2 pos forwards, and no B 1 pos back
        CDEFs = harp.index[
            (harp['Payload'] == 12) & (harp['Payload'].shift(-2) == 16) & (harp['Payload'].shift(1) != 30) & (harp['Payload'].shift(1) != 10)].tolist()

        # Every instance of G not after STOP with I 2 pos forwards.
        GHIJs = harp.index[(harp['Payload'] == 20) & (harp['Payload'].shift(-2) == 24) & (harp['Payload'].shift(1) != 30)].tolist()

        ### Testing stimuli ###
        # Every instance of E not after STOP with G 2 pos forwards, and no I 4 pos forwards.
        EFGHs = harp.index[
            (harp['Payload'] == 16) & (harp['Payload'].shift(-2) == 20) & (harp['Payload'].shift(-4) != 24) & (harp['Payload'].shift(1) != 30)].tolist()

        # Every instance of E not after STOP with H 2 pos forwards.
        EFHGs = harp.index[(harp['Payload'] == 16) & (harp['Payload'].shift(-2) == 22) & (harp['Payload'].shift(1) != 30)].tolist()

        # Every instance of B not after STOP or A.
        BCDEs = harp.index[(harp['Payload'] == 10) & (harp['Payload'].shift(1) != 30) & (harp['Payload'].shift(1) != 8)].tolist()
    
    stimuli_list = [Xs, ABCDs, CDEFs, GHIJs, EFGHs, EFHGs, BCDEs]
    for stimulus in stimuli_list:
        for index in stimulus:
            harp.at[index, 'id'] = STIMULI[stimuli_list.index(stimulus)]
    
    event_times_by_event = {}
    aligned_pupil = {}
    for event_id in STIMULI:
        event_times_by_event[event_id] = harp[harp['id']==event_id]['Timestamp'].values
        
    for event_id, event_times in event_times_by_event.items():
        epochs = [pupil.loc[t + READING_WINDOW[0]:t + READING_WINDOW[1]] for t in event_times]
        if epochs == []:
            continue
        epochs_array = np.full((len(epochs),max([len(e) for e in epochs])), np.nan)
        for index, epoch in enumerate(epochs):
            epochs_array[index][:len(epoch)] = epoch.values
        aligned_pupil[event_id] = epochs_array[-300:]

    x_ser = np.round(np.linspace(READING_WINDOW[0], READING_WINDOW[1], aligned_pupil['X'].shape[1]), 2)

    for event_id in aligned_pupil:
        if aligned_pupil[event_id].shape[1] < x_ser.shape[0]:
            aligned_pupil[event_id] = np.pad(aligned_pupil[event_id], [(0,0),(0, x_ser.shape[0] - aligned_pupil[event_id].shape[1])], mode='constant', constant_values=np.nan)
        aligned_pupil[event_id] = pd.DataFrame(aligned_pupil[event_id],columns=x_ser)
    
    aligned_pupil_by_session[session_id] = aligned_pupil


aggregated_aligned_pupil = aggregate_total(aligned_pupil_by_session)
if SHOW_DISTRIBUTION:
    plt.pause(0.1)
    dist_plot = plt.subplots()
    actual_distribution = {}
    for stimulus in STIMULI:
        actual_distribution[stimulus] = len(aggregated_aligned_pupil[stimulus])
        dist_plot[1].bar(stimulus, actual_distribution[stimulus], color=STIMULUS_COLOURS.get(stimulus, None))
        dist_plot[1].text(stimulus, actual_distribution[stimulus] + 5, str(actual_distribution[stimulus]), ha='center',
                          va='center')

    dist_plot[0].suptitle(f'Overall distribution for: {animals_to_list}')
    annotation = f'n = {aggregated_aligned_pupil["X"].shape[0]} trials'
    dist_plot[1].annotate(annotation, xy=(0.3, 1.02), xycoords=dist_plot[1].get_xaxis_transform())
    fig = plt.gcf()
    dist_plot[0].show()
    fig.savefig(
        fr'C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\Outputs\Second Patterns Only\Actual Distributions\{animals_to_list}_distribution.png'
    )
#pickle.dump(aligned_pupil_by_session, open(fr"C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\stage{STAGE}_aligned_pupil_by_session.pkl", 'wb'))

if PLOT_BY_SESSION:
    plot_by_session(aligned_pupil_by_session, True)

plt.pause(0.1)
pupil_plot = plt.subplots()
for event_id, response in aggregated_aligned_pupil.items():
    baseline_mean = response.loc[:, -1:0].mean(axis=1)
    baselined = response.sub(baseline_mean, axis=0)
    pupil_plot[1].plot(baselined.columns, baselined.mean(axis=0),label=event_id, color=STIMULUS_COLOURS.get(event_id, None))
    plot_shaded_error_ts(pupil_plot[1],baselined.columns,baselined.mean(axis=0), baselined.sem(axis=0),alpha=0.1, color=STIMULUS_COLOURS.get(event_id, None))
pupil_plot[1].legend()
pupil_plot[1].set_xlim((PLOTTING_WINDOW[0], PLOTTING_WINDOW[1]))
annotation = f'n = {aggregated_aligned_pupil["X"].shape[0]} trials'
pupil_plot[1].annotate(annotation, xy=(0.3, 1.02), xycoords=pupil_plot[1].get_xaxis_transform())
pupil_plot[1].axvspan(0, 0.15, color='grey', alpha=0.1)
pupil_plot[1].axvspan(0.5, 0.65, color='grey', alpha=0.1)
pupil_plot[1].axvspan(1, 1.15, color='grey', alpha=0.1)
pupil_plot[1].axvspan(1.5, 1.65, color='grey', alpha=0.1)
#pupil_plot[1].set_ylim((-0.5,1))
#pupil_plot[1].set_ylim((-0.25,0.75))
animals_to_list = ', '.join(ANIMALS)
pupil_plot[0].suptitle(f'Baseline subtracted plot for {animals_to_list}')
fig = plt.gcf()
pupil_plot[0].show()
fig.savefig(fr'C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\Outputs\Second Patterns Only\{animals_to_list}\{animals_to_list}_Baseline_Subtracted.png')



# --- Example Usage ---
# Session-wide baseline regression across all sessions:
baseline_window = (-1,0)
df_for_regr = build_session_wide_baseline_dataframe(
    aligned_pupil_by_session,
    include_events=[event_id for event_id in STIMULI if event_id != 'X'],
    smooth_window=9
)
coefs, r2, r2_by_feature = fit_time_resolved_baseline_regression(df_for_regr, base_window=baseline_window)

# To plot how the influence of baseline mean changes over time:
plt.pause(0.1)
base_eff_plot= plt.subplots()
for coef_name in coefs:
    print(coef_name)
    base_eff_plot[1].plot(coefs.index, coefs[coef_name], label=f'Baseline {coef_name} Effect')
base_eff_plot[1].axvline(0, color='k', linestyle='--')
base_eff_plot[1].legend()
base_eff_plot[0].suptitle(f'Baselined effects plot for {animals_to_list}')
fig = plt.gcf()
base_eff_plot[0].show()
fig.savefig(fr'C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\Outputs\Second Patterns Only\{animals_to_list}\{animals_to_list}_Coeff_Contribution.png')

# Plot R2
plt.pause(0.1)
base_r2_plot= plt.subplots()
base_r2_plot[1].plot(r2,label='Total R2')
for coef_name in r2_by_feature:
    base_r2_plot[1].plot(r2_by_feature[coef_name], label=f'R2 {coef_name}')
base_r2_plot[1].axvline(0, color='k', linestyle='--')
base_r2_plot[1].legend()
base_r2_plot[0].suptitle(f'Baselined R2 plot for {animals_to_list}')
fig = plt.gcf()
base_r2_plot[0].show()
fig.savefig(fr'C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\Outputs\Second Patterns Only\{animals_to_list}\{animals_to_list}_Coeff_R2.png')

# Apply the session-wide baseline regression back to every session.
baselined_sessions = regress_out_baseline_across_sessions(
    aligned_pupil_by_session,
    coefs[['immediate','slope','mean']],
    base_window=baseline_window,
    events_to_regress=[event_id for event_id in STIMULI if event_id != 'X']
)

aggregated_baselined = aggregate_total(baselined_sessions)

# plot training data
pupil_plot = plt.subplots()
for event_id, response in aggregated_baselined.items():
    if event_id == 'X':
        continue
    if event_id in ['EFGH', 'EFHG', 'BCDE']:
        continue
    pupil_plot[1].plot(response.columns, response.mean(axis=0), label=event_id, color=STIMULUS_COLOURS.get(event_id, None))
    plot_shaded_error_ts(pupil_plot[1], response.columns, response.mean(axis=0),
                         response.sem(axis=0), alpha=0.1, color=STIMULUS_COLOURS.get(event_id, None))
pupil_plot[1].legend()
pupil_plot[1].set_xlim((-1,4))
pupil_plot[1].axvline(0, color='k', linestyle='--')
annotation = f'n = {aggregated_baselined["X"].shape[0]} trials'
pupil_plot[1].annotate(annotation, xy=(0.3, 1.02), xycoords=pupil_plot[1].get_xaxis_transform())
pupil_plot[1].set_ylim(Y_LIMS.get(animals_to_list, (-0.5,0.5)))
pupil_plot[1].axvspan(0, 0.15, color='grey', alpha=0.1)
pupil_plot[1].axvspan(0.5, 0.65, color='grey', alpha=0.1)
pupil_plot[1].axvspan(1, 1.15, color='grey', alpha=0.1)
pupil_plot[1].axvspan(1.5, 1.65, color='grey', alpha=0.1)
#pupil_plot[0].suptitle(f'Baseline Regressed plot for {animals_to_list}')
pupil_plot[0].suptitle(f'Baseline Regressed plot of training stimuli for {animals_to_list}')
fig = plt.gcf()
pupil_plot[0].show()
fig.savefig(fr'C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\Outputs\Second Patterns Only\{animals_to_list}\{animals_to_list}_Baseline_Regressed_Training.png')
plt.pause(0.1)

# plot testing data
pupil_plot = plt.subplots()
for event_id, response in aggregated_baselined.items():
    if event_id == 'X':
        continue
    if event_id in ['ABCD', 'CDEF', 'GHIJ']:
        continue
    pupil_plot[1].plot(response.columns, response.mean(axis=0), label=event_id, color=STIMULUS_COLOURS.get(event_id, None))
    plot_shaded_error_ts(pupil_plot[1], response.columns, response.mean(axis=0),
                         response.sem(axis=0), alpha=0.1, color=STIMULUS_COLOURS.get(event_id, None))
pupil_plot[1].legend()
pupil_plot[1].set_xlim((-1,4))
pupil_plot[1].axvline(0, color='k', linestyle='--')
annotation = f'n = {aggregated_baselined["X"].shape[0]} trials'
pupil_plot[1].annotate(annotation, xy=(0.3, 1.02), xycoords=pupil_plot[1].get_xaxis_transform())
pupil_plot[1].set_ylim(Y_LIMS.get(animals_to_list, (-0.5,0.5)))
pupil_plot[1].axvspan(0, 0.15, color='grey', alpha=0.1)
pupil_plot[1].axvspan(0.5, 0.65, color='grey', alpha=0.1)
pupil_plot[1].axvspan(1, 1.15, color='grey', alpha=0.1)
pupil_plot[1].axvspan(1.5, 1.65, color='grey', alpha=0.1)
#pupil_plot[0].suptitle(f'Baseline Regressed plot for {animals_to_list}')
pupil_plot[0].suptitle(f'Baseline Regressed plot of testing stimuli for {animals_to_list}')
fig = plt.gcf()
pupil_plot[0].show()
fig.savefig(fr'C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\Outputs\Second Patterns Only\{animals_to_list}\{animals_to_list}_Baseline_Regressed_Testing.png')
plt.pause(0.1)

# plot all
pupil_plot = plt.subplots()
for event_id, response in aggregated_baselined.items():
    pupil_plot[1].plot(response.columns, response.mean(axis=0), label=event_id, color=STIMULUS_COLOURS.get(event_id, None))
    plot_shaded_error_ts(pupil_plot[1], response.columns, response.mean(axis=0),
                         response.sem(axis=0), alpha=0.1, color=STIMULUS_COLOURS.get(event_id, None))
pupil_plot[1].legend()
pupil_plot[1].set_xlim((-1,4))
pupil_plot[1].axvline(0, color='k', linestyle='--')
annotation = f'n = {aggregated_baselined["X"].shape[0]} trials'
pupil_plot[1].annotate(annotation, xy=(0.3, 1.02), xycoords=pupil_plot[1].get_xaxis_transform())
pupil_plot[1].axvspan(0, 0.15, color='grey', alpha=0.1)
pupil_plot[1].axvspan(0.5, 0.65, color='grey', alpha=0.1)
pupil_plot[1].axvspan(1, 1.15, color='grey', alpha=0.1)
pupil_plot[1].axvspan(1.5, 1.65, color='grey', alpha=0.1)
#pupil_plot[1].set_ylim(Y_LIMS.get(animals_to_list, (-0.5,0.5)))
#pupil_plot[0].suptitle(f'Baseline Regressed plot for {animals_to_list}')
pupil_plot[0].suptitle(f'Baseline Regressed plot for {animals_to_list}')
fig = plt.gcf()
pupil_plot[0].show()
fig.savefig(fr'C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\Outputs\Second Patterns Only\{animals_to_list}\{animals_to_list}_Baseline_Regressed.png')
plt.pause(0.1)

# plot training data
pupil_plot = plt.subplots()
n_stimuli = 0
for event_id, response in aggregated_aligned_pupil.items():
    if event_id == 'X':
        continue
    if event_id in ['EFGH', 'EFHG', 'BCDE']:
        continue
    n_stimuli += len(response.index)
    baseline_mean = response.loc[:, -1:0].mean(axis=1)
    baselined = response.sub(baseline_mean, axis=0)
    pupil_plot[1].plot(baselined.columns, baselined.mean(axis=0), label=event_id, color=STIMULUS_COLOURS.get(event_id, None))
    plot_shaded_error_ts(pupil_plot[1], baselined.columns, baselined.mean(axis=0),
                         baselined.sem(axis=0), alpha=0.1, color=STIMULUS_COLOURS.get(event_id, None))
pupil_plot[1].legend()
pupil_plot[1].set_xlim((-1,4))
pupil_plot[1].axvline(0, color='k', linestyle='--')
annotation = f'n = {n_stimuli} stimuli'
pupil_plot[1].annotate(annotation, xy=(0.3, 1.02), xycoords=pupil_plot[1].get_xaxis_transform())
pupil_plot[1].set_ylim(Y_LIMS.get(animals_to_list, (-0.5,0.5)))
pupil_plot[1].axvspan(0, 0.15, color='grey', alpha=0.1)
pupil_plot[1].axvspan(0.5, 0.65, color='grey', alpha=0.1)
pupil_plot[1].axvspan(1, 1.15, color='grey', alpha=0.1)
pupil_plot[1].axvspan(1.5, 1.65, color='grey', alpha=0.1)
pupil_plot[0].suptitle(f'Baseline Subtracted plot of training stimuli for {animals_to_list}')
fig = plt.gcf()
pupil_plot[0].show()
fig.savefig(fr'C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\Outputs\Second Patterns Only\{animals_to_list}\{animals_to_list}_Baseline_Subtracted_Training.png')
plt.pause(0.1)

# plot testing data
pupil_plot = plt.subplots()
n_stimuli = 0
for event_id, response in aggregated_aligned_pupil.items():
    if event_id == 'X':
        continue
    if event_id in ['ABCD', 'CDEF', 'GHIJ']:
        continue
    n_stimuli += len(response.index)
    baseline_mean = response.loc[:, -1:0].mean(axis=1)
    baselined = response.sub(baseline_mean, axis=0)
    pupil_plot[1].plot(baselined.columns, baselined.mean(axis=0), label=event_id, color=STIMULUS_COLOURS.get(event_id, None))
    plot_shaded_error_ts(pupil_plot[1], baselined.columns, baselined.mean(axis=0),
                         baselined.sem(axis=0), alpha=0.1, color=STIMULUS_COLOURS.get(event_id, None))
pupil_plot[1].legend()
pupil_plot[1].set_xlim((-1,4))
pupil_plot[1].axvline(0, color='k', linestyle='--')
annotation = f'n = {n_stimuli} stimuli'
pupil_plot[1].annotate(annotation, xy=(0.3, 1.02), xycoords=pupil_plot[1].get_xaxis_transform())
pupil_plot[1].set_ylim(Y_LIMS.get(animals_to_list, (-0.5,0.5)))
pupil_plot[1].axvspan(0, 0.15, color='grey', alpha=0.1)
pupil_plot[1].axvspan(0.5, 0.65, color='grey', alpha=0.1)
pupil_plot[1].axvspan(1, 1.15, color='grey', alpha=0.1)
pupil_plot[1].axvspan(1.5, 1.65, color='grey', alpha=0.1)
pupil_plot[0].suptitle(f'Baseline Subtracted plot of testing stimuli for {animals_to_list}')
fig = plt.gcf()
pupil_plot[0].show()
fig.savefig(fr'C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\Outputs\Second Patterns Only\{animals_to_list}\{animals_to_list}_Baseline_Subtracted_Testing.png')
plt.pause(0.1)