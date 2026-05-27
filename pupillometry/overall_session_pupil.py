import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np

STAGE = 3

pupil_df = pd.read_csv(fr"C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\stage{STAGE}_pupil_df.csv", index_col=0)

for session in pupil_df['session_id'].unique():
    pupil_sess_df = pupil_df[pupil_df['session_id'] == session]
    pupil_sess_df = pupil_sess_df.reset_index()
    pupil_sess_df['Time (min)'] = (pupil_sess_df['index'] - pupil_sess_df['index'][0]) / 60.0
    pupil_sess_df.plot(y='pupilsense_raddi_a_zscored', x='Time (min)', title=session)
    #, figsize=(12.8, 9.2)

    fig = plt.gcf()
    plt.show()
    fig.savefig(fr'C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\Outputs\Whole Session Pupils\{session}_fullsession.png')