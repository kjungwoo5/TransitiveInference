import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np

def plot_sessionwide_pupil_dilation(pupil_df: pd.DataFrame, output_path: Path, pupil_df_query = None):
    if pupil_df_query:
        pupil_df = pupil_df.query(pupil_df_query)
    for session in pupil_df['session_id'].unique():
        pupil_sess_df = pupil_df[pupil_df['session_id'] == session]
        pupil_sess_df = pupil_sess_df.reset_index()
        pupil_sess_df['Time (min)'] = (pupil_sess_df['index'] - pupil_sess_df['index'][0]) / 60.0
        pupil_sess_df.plot(y='pupilsense_raddi_a_zscored', x='Time (min)', title=session, figsize=(12.8, 9.2))

        fig = plt.gcf()
        plt.show()
        fig.savefig(output_path / fr'Whole Session Pupils\{session}_fullsession.png')
        
