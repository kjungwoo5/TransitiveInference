import statsmodels.formula.api as smf

import json

from pathlib import Path
import sys
sys.path.append('../')
sys.path.append(str(Path(__file__).parent.parent.parent))
from Analysis.XdetectionCore.xdetectioncore.decoding.decoding_funcs import Decoder
from joblib import Parallel, delayed

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import behaviour as tfb
import data_io as tfio
import pupillometry as tfp
import tqdm
from pupillometry import PupilPlotter

SESSION_PATH = Path(r"X:\Dammy\Xdetection_mouse_hf_test\session_topology_transitive_inference_cleaned.csv")
HOME_PATH = Path(r"C:\bonsai\data\JungWoo")
OUTPUT_PATH = Path(r"C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\Outputs")
# PARQUET_DIR = Path(r'X:\Dammy\mouse_pupillometry\pickles\trans_inf_test_90Hz_hpass00_lpass0')
PARQUET_DIR = Path(r'X:\Dammy\mouse_pupillometry\pickles\trans_inf_bandpass_90Hz_hpass01_lpass4')
HARP_DIR = Path(r'X:\Dammy\harpbins')

STAGE = 5
    
pupil_df = tfio.load_aggregate_pupil_df(SESSION_PATH, STAGE, PARQUET_DIR)
harp_df = tfio.load_aggregate_harp_df(SESSION_PATH, STAGE, HARP_DIR)
td_df = tfio.load_aggregate_trial_data(SESSION_PATH, HOME_PATH)


harp_filtered = tfio.filter_harp_by_successful_trials(harp_df, td_df, print_trial_lengths=False)


for animal in ['JK01', 'JK02', 'JK03', 'JK04']:
    plotter = PupilPlotter(pupil_df, harp_filtered, STAGE, 'testing', OUTPUT_PATH, [animal])
    plotter.align_pupil_by_session(filter=True)
    
    
    window_from_next_pip = 0.25
    
    tones_by_position = []
    print('Animal:', animal)
    pip_df = plotter.prep_for_decoding(window_size= 0.3, tmax = 0.49)
    
    pip_df.dropna(inplace=True)
    
    pip_df = pip_df[pip_df['stimulus_id'] != 'X']
    pip_df = pip_df[pip_df['stimulus_id'] != 'GHAB']
    pip_df = pip_df[pip_df['stimulus_id'] != 'ABGH']
    
    pip_df['first_tone'] = pip_df['stimulus_id'].str[0]
    pip_df['second_tone'] = pip_df['stimulus_id'].str[1]
    pip_df['third_tone'] = pip_df['stimulus_id'].str[2]
    pip_df['fourth_tone'] = pip_df['stimulus_id'].str[3]
    
    for index, row in pip_df.iterrows():
        tones_by_position.append({
            'pupil_dilation': row['pip1'],
            'tone': row['first_tone'],
            'tone_position': 1,
            'trial': index,
        })
        tones_by_position.append({
            'pupil_dilation': row['pip2'],
            'tone': row['second_tone'],
            'tone_position': 2,
            'trial': index,
        })
        tones_by_position.append({
            'pupil_dilation': row['pip3'],
            'tone': row['third_tone'],
            'tone_position': 3,
            'trial': index,
        })
        tones_by_position.append({
            'pupil_dilation': row['pip4'],
            'tone': row['fourth_tone'],
            'tone_position': 4,
            'trial': index,
        })
    
    df = pd.DataFrame(tones_by_position)
    df = df.dropna(subset=['pupil_dilation', 'tone', 'tone_position'])
    df['animal'] = animal

    #print(df.head())

    # 2. FIT MIXED-EFFECTS INTERACTION MODEL
    formula = "pupil_dilation ~ C(tone) * tone_position"
    model = smf.mixedlm(
        formula,
        data=df,
        groups=df['trial'],
        re_formula='1',
    )
    results = model.fit()
    print(results.summary())
    fixed_effects = results.fe_params
    print(fixed_effects)
    
    intercept = fixed_effects['Intercept']
    b_position = fixed_effects['tone_position']
    
    b_tone_D = fixed_effects['C(tone)[T.D]']
    b_tone_E = fixed_effects['C(tone)[T.E]']
    b_tone_F = fixed_effects['C(tone)[T.F]']
    
    b_int_D = fixed_effects['C(tone)[T.D]:tone_position']
    b_int_E = fixed_effects['C(tone)[T.E]:tone_position']
    b_int_F = fixed_effects['C(tone)[T.F]:tone_position']
    
    pupil_C = intercept + (b_position) * df['tone_position']
    pupil_D = (intercept + b_tone_D) + (b_position + b_int_D) * df['tone_position']
    pupil_E = (intercept + b_tone_E) + (b_position + b_int_E) * df['tone_position']
    pupil_F = (intercept + b_tone_F) + (b_position + b_int_F) * df['tone_position']
    
    plt.figure(figsize=(7, 5))

    # Plot lines for each tone
    plt.plot(df['tone_position'], pupil_C, label='C', color='#43a047', linewidth=2)
    plt.plot(df['tone_position'], pupil_D, label='D', color='#1e88e5', linewidth=2, linestyle='--')
    plt.plot(df['tone_position'], pupil_E, label='E', color='#f59e0b', linewidth=2, linestyle='-.')
    plt.plot(df['tone_position'], pupil_F, label='F', color='#7b1fa2', linewidth=2, linestyle=':')

    # Formatting
    plt.title(f'{animal} Interaction Plot: Tone x Tone Position on Pupil Dilation', fontsize=12, pad=15)
    plt.xticks(np.arange(1, 5, step=1))
    plt.xlabel('Tone Position', fontsize=10)
    plt.ylabel('Predicted Pupil Dilation', fontsize=10)
    plt.legend(title='Tone', frameon=True, loc = 1)
    plt.grid(True, linestyle=':', alpha=0.6)

    # Display plot
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    fig.savefig(OUTPUT_PATH / 'Linear Mixed Effects' / f'{animal}_interactions_plot.png')