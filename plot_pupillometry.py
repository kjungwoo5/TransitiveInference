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

pupil_df = pd.read_csv(fr"C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\stage{STAGE}_pupil_df.csv", index_col=0)

SESSION_PATH = Path(r"X:\Dammy\Xdetection_mouse_hf_test\session_topology_transitiveinference_pilot.csv")
HOME_PATH = Path(r"C:\bonsai\data\JungWoo")
OUTPUT_PATH = Path(r"C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\Outputs")
PARQUET_DIR = Path(r'X:\Dammy\mouse_pupillometry\pickles\trans_inf_test_90Hz_hpass00_lpass0')
HARP_DIR = Path(r'X:\Dammy\harpbins')
STAGE = 3

if __name__ == "__main__":
    pupil_df = tfio.load_aggregate_pupil_df(SESSION_PATH, 3, PARQUET_DIR)
    harp_df = tfio.load_aggregate_harp_df(SESSION_PATH, 3, HARP_DIR)
    
    JK01_stage3_first = PupilPlotter(pupil_df, harp_df, 3, 'first', OUTPUT_PATH, ['JK01'])