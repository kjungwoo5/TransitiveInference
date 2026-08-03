import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import permutations
from pathlib import Path
from numpy.linalg import norm
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from copy import copy
import os

import sys
sys.path.append('../')
sys.path.append(str(Path(__file__).parent.parent.parent))
from Analysis.XdetectionCore.xdetectioncore.plotting import plot_shaded_error_ts, format_axis

READING_WINDOW = [-2, 5]
PLOTTING_WINDOW = [-1, 4]


def _build_cdef_permutation_colours():
    """Create distinct but related colour shades for each CDEF permutation."""
    palette_by_start = {
        'C': ['#1b5e20', '#2e7d32', '#43a047', '#66bb6a', '#a5d6a7', '#c8e6c9'],
        'D': ['#0d47a1', '#1565c0', '#1e88e5', '#42a5f5', '#90caf9', '#bbdefb'],
        'E': ['#b26a00', '#d97706', '#f59e0b', '#fbbf24', '#fde68a', '#fef3c7'],
        'F': ['#4a148c', '#6a1b9a', '#7b1fa2', '#8e24aa', '#ce93d8', '#e1bee7'],
    }

    colours = {}
    for start_letter, palette in palette_by_start.items():
        ordered_permutations = [perm for perm in permutations('CDEF') if perm[0] == start_letter]
        for perm, colour in zip(ordered_permutations, palette):
            colours[''.join(perm)] = colour
    return colours


STIMULUS_COLOURS = {
    'X': 'k',
    'ABCD': 'c',
    'CDEF': 'm',
    'GHIJ': 'y',
    'EFGH': 'b',
    'EFHG': 'r',
    'BCDE': 'g',
    'Normal': 'b',
    'Deviant': 'r',
    'A': '#292f56',
    'B': '#4a3867',
    'C': '#6d406f',
    'D': '#904b6c',
    'E': '#af595f',
    'F': '#c66d4a',
    'G': '#d28a30',
    'H': '#cfab27',
    'I': '#bacf4e',
    'J': '#90f28c',
    'CFED': 'r'
}

OUTPUT_SUBDIRS = {
    'testing': 'Testing Phase',
    'exposure': 'Exposure Phase',
    'first': 'First Patterns Only',
    'second': 'Second Patterns Only',
}

ALL_ANIMALS = ['JK01', 'JK02', 'JK03', 'JK04']

Y_LIMS = {
    # 4: {
    #     'JK01': (-0.25,0.35),
    #     'JK02': (-0.5,0.7),
    #     'JK03': (-0.5,0.4),
    #     'JK04': (-0.5,0.6),
    #     ' JK01, JK02, JK03, JK04 ': (-0.35,0.35),
    #     'JK01_early_filtered': (-0.3, 0.3),
    #     'JK01_late_filtered': (-0.3, 0.3),
    #     'JK01_filtered': (-0.15, 0.3),
    #     'JK02_filtered': (-0.4, 0.6),
    #     'JK03_filtered': (-0.4, 0.8),
    #     'JK04_filtered': (-0.5, 0.8),
    # },
    # 5: {
    #     'JK01': (-0.25,0.35),
    #     'JK02': (-0.5,0.7),
    #     'JK03': (-0.5,0.4),
    #     'JK04': (-0.5,0.6),
    #     ' JK01, JK02, JK03, JK04 ': (-0.35,0.35),
    #     'JK01_early_filtered': (-0.3, 0.3),
    #     'JK01_late_filtered': (-0.3, 0.3),
    #     'JK01_filtered': (-0.2, 0.35),
    #     'JK02_filtered': (-0.3, 0.3),
    #     'JK03_filtered': (-0.3, 0.6),
    #     'JK04_filtered': (-0.4, 0.7),
    # }
}

PERMS_Y_LIMS = {
    'JK01_filtered': (-0.4, 0.7),
    'JK02_filtered': (-0.7, 1.1),
    'JK03_filtered': (-0.4, 1.3),
    'JK04_filtered': (-0.4, 1.0),
}

STAGE1_FREQUENCIES = {
    'A': '5275', 
    'B': '5920', 
    'C': '6646', 
    'D': '7459', 
    'E': '8373', 
    'F': '9398', 
    'G': '10550', 
    'H': '11841', 
    'I': '13292', 
    'J': '14919',
}

THESIS_STAGES = {
    1: 1,
    4: 2, 
    5: 3,
}


def autolabel(rects, ax):
    # Get y-axis height to calculate label position from.
    (y_bottom, y_top) = ax.get_ylim()
    y_height = y_top - y_bottom

    for rect in rects:
        height = rect.get_height()

        # Fit the label above the column
        label_position = height + (y_height * 0.01)

        ax.text(rect.get_x() + rect.get_width()/2., label_position,
                '%d' % int(height),
                ha='center', va='bottom')

class PupilPlotter:
    from stage_specific_funcs import plot_pitch_dependency, plot_difference, plot_cosine_similarity, prep_for_decoding, plot_stage5_perms
    from stage_specific_funcs import plot_stage1_window_diagram, plot_differences_method, plot_pupil_preprocessing, plot_stage1_peak_time_by_stimulus
    
    def __init__(self, pupil_df: pd.DataFrame, harp_df: pd.DataFrame, stage: int, type_of_analysis: str, output_path: Path, animals: list):
        valid_types_of_analysis = {'testing', 'exposure', 'first', 'second'}
        if type_of_analysis not in valid_types_of_analysis:
            raise Exception('Not a valid type of analysis! (\'testing\', \'exposure\', \'first\', \'second\')')
        if type_of_analysis in {'first', 'second'} and stage < 3:
            raise Exception(f'{type_of_analysis} is not valid for stage {stage}!')
        

        self.stage = stage
        if stage == 5:
            STIMULUS_COLOURS.update(_build_cdef_permutation_colours())
        
        self.type_of_analysis = type_of_analysis
        self.output_path = output_path
        self.animals = animals
        animals_to_drop = set(self.animals) ^ set(ALL_ANIMALS)
        for animal in animals_to_drop:
            pupil_df = pupil_df[pupil_df['session_id'].str.contains(animal) == False]
            harp_df = harp_df[harp_df['session_id'].str.contains(animal) == False]
        self.pupil_df = pupil_df
        self.harp_df = harp_df
        self.output_subdir = OUTPUT_SUBDIRS.get(type_of_analysis, None)
        if not self.output_subdir:
            raise Exception(f'Something went wrong. Type of analysis = {self.type_of_analysis}')
        
        
    def set_early_sessions(self, proportion = 0.4):
        sessions = list(self.pupil_df['session_id'].unique())
        early_sessions = sessions[:int(len(sessions) * proportion)]
        self.harp_df = self.harp_df[self.harp_df['session_id'].isin(early_sessions)]
        self.pupil_df = self.pupil_df[self.pupil_df['session_id'].isin(early_sessions)]
        self.animals = [a + '_early' for a in self.animals]
    
    def set_late_sessions(self, proportion = 0.4):
        sessions = list(self.pupil_df['session_id'].unique())
        late_sessions = sessions[int(len(sessions) * proportion):]
        self.harp_df = self.harp_df[self.harp_df['session_id'].isin(late_sessions)]
        self.pupil_df = self.pupil_df[self.pupil_df['session_id'].isin(late_sessions)]
        self.animals = [a + '_late' for a in self.animals]
    
    def get_stimuli(self, harp, filter):
        if self.type_of_analysis != 'exposure':
            # Take harp data only past the first 100 trials (i.e. occurrences of X)
            if self.stage != 5: 
                Xs = harp.index[harp['Payload'] == 3].tolist()
                if len(Xs) > 100:
                    harp = harp[harp.index >= Xs[100]]
                    Xs = Xs[100:]
            else: 
                Xs = harp.index[harp['Payload'] == 3].tolist()
            if filter: 
                harp = harp[harp['Outcome'] == 1]
                if '_filtered' not in ''.join(self.animals):
                    self.animals = [a + '_filtered' for a in self.animals]
                Xs = harp.index[harp['Payload'] == 3].tolist()
            if self.stage == 1:
                types_of_stimuli = ['X', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
                As = harp.index[harp['Payload'] == 8].tolist()
                Bs = harp.index[harp['Payload'] == 10].tolist()
                Cs = harp.index[harp['Payload'] == 12].tolist()
                Ds = harp.index[harp['Payload'] == 14].tolist()
                Es = harp.index[harp['Payload'] == 16].tolist()
                Fs = harp.index[harp['Payload'] == 18].tolist()
                Gs = harp.index[harp['Payload'] == 20].tolist()
                Hs = harp.index[harp['Payload'] == 22].tolist()
                Is = harp.index[harp['Payload'] == 24].tolist()
                Js = harp.index[harp['Payload'] == 26].tolist()
                
                stimuli_list = [Xs, As, Bs, Cs, Ds, Es, Fs, Gs, Hs, Is, Js]
            
            elif self.stage == 2:
                types_of_stimuli = ['X', 'Normal', 'Deviant']
                normals = harp.index[(harp['Payload'] == 25) & (harp['Payload'].shift(-2) == 27)].tolist()
                deviants = harp.index[(harp['Payload'] == 25) & (harp['Payload'].shift(-2) == 25)].tolist()
        
                stimuli_list = [Xs, normals, deviants]
                
                
            elif self.stage == 3:
                types_of_stimuli = ['X', 'ABCD', 'CDEF', 'GHIJ', 'EFGH', 'EFHG', 'BCDE']
                if self.type_of_analysis == 'testing':
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
                
                elif self.type_of_analysis == 'first':
                    ### Training stimuli ###
                    # Every instance of A after STOP
                    ABCDs = harp.index[(harp['Payload'] == 8) & (harp['Payload'].shift(1) == 30)].tolist()
                    # Every instance of C after STOP with E 2 pos forwards.
                    CDEFs = harp.index[(harp['Payload'] == 12) & (harp['Payload'].shift(-2) == 16) & (harp['Payload'].shift(1) == 30)].tolist()
                    # Every instance of G after STOP with I 2 pos forwards.
                    GHIJs = harp.index[(harp['Payload'] == 20) & (harp['Payload'].shift(-2) == 24) & (harp['Payload'].shift(1) == 30)].tolist()

                    ### Testing stimuli ###
                    # Every instance of E after STOP with G 2 pos forwards, and no I 4 pos forwards.
                    EFGHs = harp.index[(harp['Payload'] == 16) & (harp['Payload'].shift(-2) == 20) & (harp['Payload'].shift(-4) != 24) & (harp['Payload'].shift(1) == 30)].tolist()
                    # Every instance of E after STOP with H 2 pos forwards.
                    EFHGs = harp.index[(harp['Payload'] == 16) & (harp['Payload'].shift(-2) == 22) & (harp['Payload'].shift(1) == 30)].tolist()
                    # Every instance of B after STOP.
                    BCDEs = harp.index[(harp['Payload'] == 10) & (harp['Payload'].shift(1) == 30)].tolist()
                
                elif self.type_of_analysis == 'second':
                    ### Training stimuli ###
                    # Every instance of A not after STOP
                    ABCDs = harp.index[(harp['Payload'] == 8) & (harp['Payload'].shift(1) != 30)].tolist()
                    # Every instance of C not after STOP with E 2 pos forwards, and no B 1 pos back
                    CDEFs = harp.index[(harp['Payload'] == 12) & (harp['Payload'].shift(-2) == 16) & (harp['Payload'].shift(1) != 30) & (harp['Payload'].shift(1) != 10)].tolist()
                    # Every instance of G not after STOP with I 2 pos forwards.
                    GHIJs = harp.index[(harp['Payload'] == 20) & (harp['Payload'].shift(-2) == 24) & (harp['Payload'].shift(1) != 30)].tolist()

                    ### Testing stimuli ###
                    # Every instance of E not after STOP with G 2 pos forwards, and no I 4 pos forwards.
                    EFGHs = harp.index[(harp['Payload'] == 16) & (harp['Payload'].shift(-2) == 20) & (harp['Payload'].shift(-4) != 24) & (harp['Payload'].shift(1) != 30)].tolist()
                    # Every instance of E not after STOP with H 2 pos forwards.
                    EFHGs = harp.index[(harp['Payload'] == 16) & (harp['Payload'].shift(-2) == 22) & (harp['Payload'].shift(1) != 30)].tolist()
                    # Every instance of B not after STOP or A.
                    BCDEs = harp.index[(harp['Payload'] == 10) & (harp['Payload'].shift(1) != 30) & (harp['Payload'].shift(1) != 8)].tolist()
                
                stimuli_list = [Xs, ABCDs, CDEFs, GHIJs, EFGHs, EFHGs, BCDEs]
            
            elif self.stage == 4:
                types_of_stimuli = ['X', 'ABCD', 'EFGH', 'CDEF', 'CFED']
                if self.type_of_analysis == 'testing':
                    ### Training stimuli ###
                    # Every instance of A
                    ABCDs = harp.index[harp['Payload'] == 10].tolist()
                    # Every instance of E with G 2 pos forwards.
                    EFGHs = harp.index[(harp['Payload'] == 18) & (harp['Payload'].shift(-2) == 22)].tolist()

                    ### Testing stimuli ###
                    # Every instance of C with E 2 pos forwards, and no B 1 pos back.
                    CDEFs = harp.index[(harp['Payload'] == 14) & (harp['Payload'].shift(-2) == 18) & (harp['Payload'].shift(1) != 12)].tolist()
                    # Every instance of C with F 1 pos forwards, and no B 1 pos back.
                    CFEDs = harp.index[(harp['Payload'] == 14) & (harp['Payload'].shift(-1) == 20) & (harp['Payload'].shift(1) != 12)].tolist()

                
                elif self.type_of_analysis == 'first':
                    ### Training stimuli ###
                    # Every instance of A after STOP
                    ABCDs = harp.index[(harp['Payload'] == 10) & (harp['Payload'].shift(1) == 30)].tolist()
                    # Every instance of E with G 2 pos forwards after STOP.
                    EFGHs = harp.index[(harp['Payload'] == 18) & (harp['Payload'].shift(-2) == 22) & (harp['Payload'].shift(1) == 30)].tolist()

                    ### Testing stimuli ###
                    # Every instance of C with E 2 pos forwards, and no B 1 pos back after STOP.
                    CDEFs = harp.index[(harp['Payload'] == 14) & (harp['Payload'].shift(-2) == 18) & (harp['Payload'].shift(1) != 12) & (harp['Payload'].shift(1) == 30)].tolist()
                    # Every instance of C with F 1 pos forwards, and no B 1 pos back after STOP. 
                    CFEDs = harp.index[(harp['Payload'] == 14) & (harp['Payload'].shift(-1) == 20) & (harp['Payload'].shift(1) != 12) & (harp['Payload'].shift(1) == 30)].tolist()
                    
                
                elif self.type_of_analysis == 'second':
                    ### Training stimuli ###
                    # Every instance of A not after STOP
                    ABCDs = harp.index[(harp['Payload'] == 10) & (harp['Payload'].shift(1) != 30)].tolist()
                    # Every instance of E with G 2 pos forwards not after STOP.
                    EFGHs = harp.index[(harp['Payload'] == 18) & (harp['Payload'].shift(-2) == 22) & (harp['Payload'].shift(1) != 30)].tolist()

                    ### Testing stimuli ###
                    # Every instance of C with E 2 pos forwards, and no B 1 pos back not after STOP.
                    CDEFs = harp.index[(harp['Payload'] == 14) & (harp['Payload'].shift(-2) == 18) & (harp['Payload'].shift(1) != 12) & (harp['Payload'].shift(1) != 30)].tolist()
                    # Every instance of C with F 1 pos forwards, and no B 1 pos back not after STOP. 
                    CFEDs = harp.index[(harp['Payload'] == 14) & (harp['Payload'].shift(-1) == 20) & (harp['Payload'].shift(1) != 12) & (harp['Payload'].shift(1) != 30)].tolist()
                
                stimuli_list = [Xs, ABCDs, EFGHs, CDEFs, CFEDs]
            
            elif self.stage == 5:
                tonemap = {'A': 10, 'B': 12, 'C': 14, 'D': 16, 'E': 18, 'F':20, 'G': 22, 'H': 24}
                types_of_stimuli = ['X']
                perm_CDEF = list(permutations('CDEF'))
                for perm in perm_CDEF: 
                    types_of_stimuli.append(''.join(perm))
                types_of_stimuli.append('ABGH')
                types_of_stimuli.append('GHAB')
                
                stimuli_list = [Xs]
                if self.type_of_analysis == 'testing':
                    for stimulus in types_of_stimuli[1:]:
                        sequences = harp.index[(harp['Payload'] == tonemap[stimulus[0]]) & 
                                               (harp['Payload'].shift(-1) == tonemap[stimulus[1]]) & 
                                               (harp['Payload'].shift(-2) == tonemap[stimulus[2]]) & 
                                               (harp['Payload'].shift(-3) == tonemap[stimulus[3]])].tolist()
                        stimuli_list.append(sequences)
                
                elif self.type_of_analysis == 'first':
                    for stimulus in types_of_stimuli[1:]:
                        sequences = harp.index[(harp['Payload'] == tonemap[stimulus[0]]) & 
                                               (harp['Payload'].shift(-1) == tonemap[stimulus[1]]) & 
                                               (harp['Payload'].shift(-2) == tonemap[stimulus[2]]) & 
                                               (harp['Payload'].shift(-3) == tonemap[stimulus[3]]) &
                                               (harp['Payload'].shift(1) == 30)].tolist()
                        stimuli_list.append(sequences)
                
                elif self.type_of_analysis == 'second':
                    for stimulus in types_of_stimuli[1:]:
                        sequences = harp.index[(harp['Payload'] == tonemap[stimulus[0]]) & 
                                               (harp['Payload'].shift(-1) == tonemap[stimulus[1]]) & 
                                               (harp['Payload'].shift(-2) == tonemap[stimulus[2]]) & 
                                               (harp['Payload'].shift(-3) == tonemap[stimulus[3]]) &
                                               (harp['Payload'].shift(1) != 30)].tolist()
                        stimuli_list.append(sequences)
                
        
        elif self.type_of_analysis == 'exposure':
            # Take harp data only until the first 100 trials (i.e. occurrences of X)
            Xs = harp.index[harp['Payload'] == 3].tolist()
            if len(Xs) > 100:
                Xs = Xs[:100]
                harp = harp[harp.index < Xs[100]]
            if filter: 
                harp = harp[harp['Outcome'] == 1]
                if 'filtered' not in ''.join(self.animals):
                    self.animals = [a + '_filtered' for a in self.animals]
            if self.stage == 2:
                types_of_stimuli = ['X', 'Normal']
                normals = harp.index[(harp['Payload'] == 25) & (harp['Payload'].shift(-2) == 27)].tolist()
        
                stimuli_list = [Xs, normals]
            
            elif self.stage == 3:
                types_of_stimuli = ['X', 'ABCD', 'CDEF', 'GHIJ']

                ### Training stimuli ###
                # Every instance of A
                ABCDs = harp.index[harp['Payload'] == 8].tolist()
                # Every instance of C with E 2 pos forwards, and no A 2 pos back.
                CDEFs = harp.index[(harp['Payload'] == 12) & (harp['Payload'].shift(-2) == 16) & (harp['Payload'].shift(2) != 8)].tolist()
                # Every instance of G with I 2 pos forwards. 
                GHIJs = harp.index[(harp['Payload'] == 20) & (harp['Payload'].shift(-2) == 24)].tolist()

                stimuli_list = [Xs, ABCDs, CDEFs, GHIJs]
            
            elif self.stage == 4:
                types_of_stimuli = ['X', 'ABCD', 'EFGH']

                ### Training stimuli ###
                # Every instance of A
                ABCDs = harp.index[harp['Payload'] == 10].tolist()
                # Every instance of E with G 2 pos forwards.
                EFGHs = harp.index[(harp['Payload'] == 18) & (harp['Payload'].shift(-2) == 22)].tolist()

                stimuli_list = [Xs, ABCDs, EFGHs]
        
        if not stimuli_list:
            raise Exception(f'Something went wrong. Stimuli list = {stimuli_list}')
        
        self.types_of_stimuli = types_of_stimuli
        return stimuli_list
    
    # Returns a dictionary of aligned pupil data by session by type of stimulus, and returns types of stimuli for future plotting
    def align_pupil_by_session(self, filter = False):
        session_ids = self.harp_df['session_id'].unique()
        aligned_pupil_by_session = {}
        for session_id in session_ids:
            pupil = self.pupil_df[self.pupil_df['session_id'] == session_id]['pupilsense_raddi_a_zscored']
            harp = self.harp_df[self.harp_df['session_id'] == session_id]

            stimuli_list = self.get_stimuli(harp, filter)
            
            for stimulus in stimuli_list:
                for index in stimulus:
                    harp.at[index, 'id'] = self.types_of_stimuli[stimuli_list.index(stimulus)]
            
            event_times_by_event = {}
            aligned_pupil = {}
            for event_id in self.types_of_stimuli:
                event_times_by_event[event_id] = harp[harp['id']==event_id]['Timestamp'].values
                
            for event_id, event_times in event_times_by_event.items():
                if self.stage == 1:
                    # epochs = [pupil.loc[t -0.5 :t + 2] for t in event_times]
                    epochs = [pupil.loc[t + READING_WINDOW[0]:t + READING_WINDOW[1]] for t in event_times]
                else:
                    epochs = [pupil.loc[t + READING_WINDOW[0]:t + READING_WINDOW[1]] for t in event_times]
                if epochs == []:
                    continue
                epochs_array = np.full((len(epochs),max([len(e) for e in epochs])), np.nan)
                for index, epoch in enumerate(epochs):
                    epochs_array[index][:len(epoch)] = epoch.values
                aligned_pupil[event_id] = epochs_array[-300:]

            if self.stage == 1:
                # x_ser = np.round(np.linspace(-0.5, 2, aligned_pupil['X'].shape[1]), 2)
                x_ser = np.round(np.linspace(READING_WINDOW[0], READING_WINDOW[1], aligned_pupil['X'].shape[1]), 2)
            else:
                x_ser = np.round(np.linspace(READING_WINDOW[0], READING_WINDOW[1], aligned_pupil['X'].shape[1]), 2)

            for event_id in aligned_pupil:
                if aligned_pupil[event_id].shape[1] < x_ser.shape[0]:
                    aligned_pupil[event_id] = np.pad(aligned_pupil[event_id], [(0,0),(0, x_ser.shape[0] - aligned_pupil[event_id].shape[1])], mode='constant', constant_values=np.nan)
                aligned_pupil[event_id] = pd.DataFrame(aligned_pupil[event_id],columns=x_ser)
            
            aligned_pupil_by_session[session_id] = aligned_pupil
        
        self.aligned_pupil_by_session = aligned_pupil_by_session


    def plot_sessionwide_pupil_dilation(self, pupil_df_query = None, save_figure = True, show_plot = True):
        if pupil_df_query:
            self.pupil_df = self.pupil_df.query(pupil_df_query)
        for session in self.pupil_df['session_id'].unique():
            pupil_sess_df = self.pupil_df[self.pupil_df['session_id'] == session]
            pupil_sess_df = pupil_sess_df.reset_index()
            pupil_sess_df['Time (min)'] = (pupil_sess_df['index'] - pupil_sess_df['index'][0]) / 60.0
            pupil_sess_df.plot(y='pupilsense_raddi_a_zscored', x='Time (min)', title=session, figsize=(12.8, 9.2))

            fig = plt.gcf()
            if show_plot:
                plt.show()
            if save_figure:
                os.makedirs(self.output_path / fr'Whole Session Pupils', exist_ok=True)
                fig.savefig(self.output_path / fr'Whole Session Pupils\{session}_fullsession.svg')
            fig.clf()
            plt.close()
            

    def plot_baseline_sub_aligned_pupil_by_session(self, save_figure = True, show_plot = True):
        
        valid_types_of_analysis = {'testing', 'exposure', 'first', 'second'}
        if self.type_of_analysis not in valid_types_of_analysis:
            raise Exception('Not a valid type of analysis! (\'testing\', \'exposure\', \'first\', \'second\')')

        for session, value in self.aligned_pupil_by_session.items():
            plt.pause(0.1)
            pupil_plot = plt.subplots()
            print('Plotting baseline subtracted plot for session: ', session)

            total_responses = {}
            for stimulus in self.types_of_stimuli:
                aggregate = []
                for key, value in self.aligned_pupil_by_session[session].items():
                    if stimulus == key:
                        aggregate.append(self.aligned_pupil_by_session[session][stimulus])
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
            if show_plot:
                pupil_plot[0].show()
            if save_figure:
                os.makedirs(fr'{self.output_path}\{self.output_subdir}\Individual Sessions', exist_ok=True)
                fig.savefig(
                    fr'{self.output_path}\{self.output_subdir}\Individual Sessions\Stage{self.stage}_{session}.svg')
            fig.clf()


    def plot_distribution_by_session(self, save_figure = True, show_plot = True):
        for session, value in self.aligned_pupil_by_session.items():
            total_responses = {}
            for stimulus in self.types_of_stimuli:
                aggregate = []
                for key, value in self.aligned_pupil_by_session[session].items():
                    if stimulus == key:
                        aggregate.append(self.aligned_pupil_by_session[session][stimulus])
                if aggregate:
                    total_responses[stimulus] = pd.concat(aggregate, axis=0, ignore_index=True)
                    total_responses[stimulus] = total_responses[stimulus].tail(300)

            plt.pause(0.1)
            dist_plot = plt.subplots()
            print('Plotting distribution for session: ', session)
            actual_distribution = {}
            for stimulus in self.types_of_stimuli:
                if self.stage == 5 and stimulus == 'X':
                    continue
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
            if show_plot:
                dist_plot[0].show()
            if save_figure:
                os.makedirs(fr'{self.output_path}\{self.output_subdir}\Actual Distributions', exist_ok=True)
                fig.savefig(
                    fr'{self.output_path}\{self.output_subdir}\Actual Distributions\By Session\Stage{self.stage}_{session}_distribution.svg'
                )
            fig.clf()

    def aggregate_total(self, baseline_data = False) -> dict:
        total_responses = {}
        for stimulus in self.types_of_stimuli:
            aggregate = []
            for key, value in self.aligned_pupil_by_session.items():
                if stimulus in self.aligned_pupil_by_session[key]:
                    aggregate.append(self.aligned_pupil_by_session[key][stimulus])
            total_responses[stimulus] = pd.concat(aggregate, axis=0, ignore_index=True)
        if baseline_data: 
            for event_id, response in total_responses.items():
                if self.stage == 1:
                    baseline_mean = response.loc[:, -0.10:0.10].mean(axis=1)
                else:
                    baseline_mean = response.loc[:, -0.15:0.15].mean(axis=1)
                total_responses[event_id] = response.sub(baseline_mean, axis=0)
        return total_responses

    
    def plot_overall_baseline_sub_aligned_pupil(self, save_figure = True, show_plot = True, use_median = False):
        animals_to_list = ', '.join(self.animals)

        aggregated_aligned_pupil = self.aggregate_total()
        pupil_plot = plt.subplots()
        for event_id, response in aggregated_aligned_pupil.items():
            if self.stage == 1:
                baseline_mean = response.loc[:, -0.15:0.15].mean(axis=1)
            else:
                baseline_mean = response.loc[:, -1:0].mean(axis=1)
            baselined = response.sub(baseline_mean, axis=0)
            if not use_median:
                pupil_plot[1].plot(baselined.columns, baselined.mean(axis=0),label=event_id, color=STIMULUS_COLOURS.get(event_id, None))
                plot_shaded_error_ts(pupil_plot[1],baselined.columns,baselined.mean(axis=0), baselined.sem(axis=0),alpha=0.1, color=STIMULUS_COLOURS.get(event_id, None))
            else: 
                pupil_plot[1].plot(baselined.columns, baselined.median(axis=0),label=event_id, color=STIMULUS_COLOURS.get(event_id, None))
                plot_shaded_error_ts(pupil_plot[1],baselined.columns,baselined.median(axis=0), baselined.sem(axis=0),alpha=0.1, color=STIMULUS_COLOURS.get(event_id, None))
        pupil_plot[1].legend()
        pupil_plot[1].set_xlim((PLOTTING_WINDOW[0], PLOTTING_WINDOW[1]))
        annotation = f'n = {aggregated_aligned_pupil["X"].shape[0]} trials'
        pupil_plot[1].annotate(annotation, xy=(0.3, 1.02), xycoords=pupil_plot[1].get_xaxis_transform())
        pupil_plot[1].set_ylim(Y_LIMS.get(self.stage, {}).get(animals_to_list, None))
        pupil_plot[1].axvspan(0, 0.15, color='grey', alpha=0.1)
        pupil_plot[1].axvspan(0.5, 0.65, color='grey', alpha=0.1)
        pupil_plot[1].axvspan(1, 1.15, color='grey', alpha=0.1)
        pupil_plot[1].axvspan(1.5, 1.65, color='grey', alpha=0.1)
        pupil_plot[1].set_ylabel('Pupil size (a.u.)')
        pupil_plot[1].set_xlabel('Time from stimulus onset (s)')
        pupil_plot[0].suptitle(f'{animals_to_list} pupil size for stage {THESIS_STAGES.get(self.stage)}')
        fig = plt.gcf()
        if show_plot:
            pupil_plot[0].show()
        if save_figure:
            os.makedirs(fr'{self.output_path}\{self.output_subdir}\{animals_to_list}', exist_ok=True)
            if not use_median:
                fig.savefig(fr'{self.output_path}\{self.output_subdir}\{animals_to_list}\Stage{self.stage}_{animals_to_list}_Baseline_Subtracted.svg')
            else: 
                fig.savefig(fr'{self.output_path}\{self.output_subdir}\{animals_to_list}\Stage{self.stage}_{animals_to_list}_Baseline_Subtracted_median.svg')
        fig.clf()
        
        
    
    def plot_baseline_sub_training(self, save_figure = True, show_plot = True):
        if self.stage == 2: 
            testing = ['Deviant']
        elif self.stage == 3: 
            testing = ['EFGH', 'EFHG', 'BCDE']
        elif self.stage == 4:
            testing = ['CDEF', 'CFED']
        animals_to_list = ', '.join(self.animals)
        pupil_plot = plt.subplots()
        n_stimuli = 0
        aggregated_aligned_pupil = self.aggregate_total()
        for event_id, response in aggregated_aligned_pupil.items():
            if event_id == 'X':
                continue
            if event_id in testing:
                continue
            n_stimuli += len(response.index)
            baseline_mean = response.loc[:, -1:0].mean(axis=1)
            baselined = response.sub(baseline_mean, axis=0)
            pupil_plot[1].plot(baselined.columns, baselined.mean(axis=0), label=event_id, color=STIMULUS_COLOURS.get(event_id, None))
            plot_shaded_error_ts(pupil_plot[1], baselined.columns, baselined.mean(axis=0),
                                baselined.sem(axis=0), alpha=0.1, color=STIMULUS_COLOURS.get(event_id, None))
        pupil_plot[1].legend()
        pupil_plot[1].set_xlim((-1,4))
        # pupil_plot[1].axvline(0, color='k', linestyle='--')
        annotation = f'n = {n_stimuli} stimuli'
        pupil_plot[1].annotate(annotation, xy=(0.3, 1.02), xycoords=pupil_plot[1].get_xaxis_transform())
        pupil_plot[1].set_ylim(Y_LIMS.get(self.stage, {}).get(animals_to_list, None))
        pupil_plot[1].axvspan(0, 0.15, color='grey', alpha=0.1)
        pupil_plot[1].axvspan(0.5, 0.65, color='grey', alpha=0.1)
        pupil_plot[1].axvspan(1, 1.15, color='grey', alpha=0.1)
        pupil_plot[1].axvspan(1.5, 1.65, color='grey', alpha=0.1)
        pupil_plot[1].set_ylabel('Pupil size (a.u.)')
        pupil_plot[1].set_xlabel('Time from stimulus onset (s)')
        pupil_plot[0].suptitle(f'{animals_to_list} pupil responses to training stimuli')
        fig = plt.gcf()
        if show_plot:
            pupil_plot[0].show()
        if save_figure:
            os.makedirs(fr'{self.output_path}\{self.output_subdir}\{animals_to_list}', exist_ok=True)
            fig.savefig(fr'{self.output_path}\{self.output_subdir}\{animals_to_list}\Stage{self.stage}_{animals_to_list}_Baseline_Subtracted_Training.svg')
        plt.pause(0.1)

    def plot_baseline_sub_testing(self, save_figure = True, show_plot = True):
        if self.stage == 2: 
            training = ['Normal']
        elif self.stage == 3: 
            training = ['ABCD', 'CDEF', 'GHIJ']
        elif self.stage == 4:
            training = ['ABCD', 'EFGH']
        animals_to_list = ', '.join(self.animals)
        pupil_plot = plt.subplots()
        n_stimuli = 0
        aggregated_aligned_pupil = self.aggregate_total()
        for event_id, response in aggregated_aligned_pupil.items():
            if event_id == 'X':
                continue
            if event_id in training:
                continue
            n_stimuli += len(response.index)
            baseline_mean = response.loc[:, -1:0].mean(axis=1)
            baselined = response.sub(baseline_mean, axis=0)
            pupil_plot[1].plot(baselined.columns, baselined.mean(axis=0), label=event_id, color=STIMULUS_COLOURS.get(event_id, None))
            plot_shaded_error_ts(pupil_plot[1], baselined.columns, baselined.mean(axis=0),
                                baselined.sem(axis=0), alpha=0.1, color=STIMULUS_COLOURS.get(event_id, None))
        pupil_plot[1].legend()
        pupil_plot[1].set_xlim((-1,4))
        # pupil_plot[1].axvline(0, color='k', linestyle='--')
        annotation = f'n = {n_stimuli} stimuli'
        pupil_plot[1].annotate(annotation, xy=(0.3, 1.02), xycoords=pupil_plot[1].get_xaxis_transform())
        pupil_plot[1].set_ylim(Y_LIMS.get(self.stage, {}).get(animals_to_list, None))
        pupil_plot[1].axvspan(0, 0.15, color='grey', alpha=0.1)
        pupil_plot[1].axvspan(0.5, 0.65, color='grey', alpha=0.1)
        pupil_plot[1].axvspan(1, 1.15, color='grey', alpha=0.1)
        pupil_plot[1].axvspan(1.5, 1.65, color='grey', alpha=0.1)
        pupil_plot[1].set_ylabel('Pupil size (a.u.)')
        pupil_plot[1].set_xlabel('Time from stimulus onset (s)')
        pupil_plot[0].suptitle(f'{animals_to_list} pupil responses to testing stimuli')
        fig = plt.gcf()
        if show_plot:
            pupil_plot[0].show()
        if save_figure:
            os.makedirs(fr'{self.output_path}\{self.output_subdir}\{animals_to_list}', exist_ok=True)
            fig.savefig(fr'{self.output_path}\{self.output_subdir}\{animals_to_list}\Stage{self.stage}_{animals_to_list}_Baseline_Subtracted_Testing.svg')
        fig.clf()
        
        

    def plot_overall_distribution(self, save_figure = True, show_plot = True):
        
        animals_to_list = ', '.join(self.animals)
        if animals_to_list.strip() == 'JK01_filtered, JK02_filtered, JK03_filtered, JK04_filtered':
            animals_to_list = 'all animals'
        animals_to_list = animals_to_list.strip('_filtered')
        
        aggregated_aligned_pupil = self.aggregate_total()
        plt.pause(0.1)
        if self.stage == 5:
            dist_plot = plt.subplots(figsize=(14,5.4))
        else: 
            dist_plot = plt.subplots()
        actual_distribution = {}
        y_max = 0.0
        n_stims = 0
        for stimulus in self.types_of_stimuli:
            if self.stage == 5 and (stimulus == 'ABGH' or stimulus == 'GHAB') or stimulus == 'X': 
                continue
            actual_distribution[stimulus] = len(aggregated_aligned_pupil[stimulus])
            if dist_plot[1].get_ylim()[1] * 1.035 > y_max:
                y_max = dist_plot[1].get_ylim()[1] * 1.035
                rects = dist_plot[1].bar(n_stims, actual_distribution[stimulus], color=STIMULUS_COLOURS.get(stimulus, None))
            rects = dist_plot[1].bar(stimulus, actual_distribution[stimulus], color=STIMULUS_COLOURS.get(stimulus, None))
            autolabel(rects, dist_plot[1])
            n_stims += 1
            # dist_plot[1].text(stimulus, actual_distribution[stimulus] + 5, str(actual_distribution[stimulus]), ha='center',
            #                 va='center')

        if self.stage == 1:
            xticks = np.arange(0, n_stims, 1)
            dist_plot[1].set_xticks(xticks, STAGE1_FREQUENCIES.values())
            xlabel = 'Stimuli (Hz)'
        else: 
            xlabel = 'Stimuli'
        dist_plot[0].suptitle(f'Overall distribution of stimuli for {animals_to_list}')
        annotation = f'n = {aggregated_aligned_pupil["X"].shape[0]} trials'
        dist_plot[1].set_ylim(ymax= y_max)
        dist_plot[1].set_ylabel('Count')
        dist_plot[1].set_xlabel(xlabel)
        dist_plot[1].annotate(annotation, xy=(0.005, 1.02), xycoords=dist_plot[1].get_xaxis_transform())
        fig = plt.gcf()
        if show_plot:
            dist_plot[0].show()
        if save_figure:
            os.makedirs(fr'{self.output_path}\{self.output_subdir}\Actual Distributions', exist_ok=True)
            fig.savefig(fr'{self.output_path}\{self.output_subdir}\Actual Distributions\Stage{self.stage}_{animals_to_list}_distribution.svg')
        fig.clf()
    
        
    def fit_time_resolved_baseline_regression(self, df:pd.DataFrame, base_window=(-1.0, 0.0)):
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


    def regress_out_baseline(self, df, coef_df, base_window=(-1.0, 0.0),intercept=None):
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
    
    def build_session_wide_baseline_dataframe(self, aligned_pupil_by_session: dict, include_events=None, smooth_window: int = 9):
        # Aggregate aligned session responses into one trial-by-time DataFrame.
        if include_events is None:
            include_events = self.types_of_stimuli

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


    def fit_time_resolved_baseline_regression_across_sessions(self, aligned_pupil_by_session: dict,base_window=(-1.0, 0.0),include_events=None,smooth_window: int = 9):
        df_for_regr = self.build_session_wide_baseline_dataframe(
            aligned_pupil_by_session,
            include_events=include_events,
            smooth_window=smooth_window
        )
        return self.fit_time_resolved_baseline_regression(df_for_regr, base_window=base_window)


    def regress_out_baseline_across_sessions(self, aligned_pupil_by_session: dict,
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
                    baselined_sessions[session][event_id] = self.regress_out_baseline(
                        copy(response), coef_df,
                        base_window=base_window,
                        intercept=intercept
                    )
                else:
                    baselined_sessions[session][event_id] = response.copy()
        return baselined_sessions

        
    def plot_overall_baseline_regressed_pupil(self, save_figure = True, show_plot = True, use_median = False):
        # Start using baseline regression
        
        animals_to_list = ', '.join(self.animals)
        
        baseline_window = (-1,0)
        df_for_regr = self.build_session_wide_baseline_dataframe(
            self.aligned_pupil_by_session,
            include_events=[event_id for event_id in self.types_of_stimuli if event_id != 'X'],
            smooth_window=9
        )
        coefs, r2, r2_by_feature = self.fit_time_resolved_baseline_regression(df_for_regr, base_window=baseline_window)

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
        fig.savefig(self.output_path / fr'Stage {self.stage} Baseline Regression\{animals_to_list}_Coeff_Contribution.svg')

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
        fig.savefig(self.output_path / fr'Stage {self.stage} Baseline Regression\{animals_to_list}_Coeff_R2.svg')

        # Apply the session-wide baseline regression back to every session.
        baselined_sessions = self.regress_out_baseline_across_sessions(
            self.aligned_pupil_by_session,
            coefs[['immediate','slope','mean']],
            base_window=baseline_window,
            events_to_regress=[event_id for event_id in self.types_of_stimuli if event_id != 'X']
        )
        
        total_responses = {}
        for stimulus in self.types_of_stimuli:
            aggregate = []
            for key, value in baselined_sessions.items():
                if stimulus in baselined_sessions[key]:
                    aggregate.append(baselined_sessions[key][stimulus])
            total_responses[stimulus] = pd.concat(aggregate, axis=0, ignore_index=True)
        
        aggregated_baselined = total_responses

        # plot all
        pupil_plot = plt.subplots()
        for event_id, response in aggregated_baselined.items():
            baseline_mean = response.loc[:, -1:0].mean(axis=1)
            baselined = response.sub(baseline_mean, axis=0)
            pupil_plot[1].plot(baselined.columns, baselined.mean(axis=0), label=event_id, color=STIMULUS_COLOURS.get(event_id, None))
            plot_shaded_error_ts(pupil_plot[1], baselined.columns, baselined.mean(axis=0),
                                baselined.sem(axis=0), alpha=0.1, color=STIMULUS_COLOURS.get(event_id, None))
        pupil_plot[1].legend()
        pupil_plot[1].set_xlim((-1,4))
        annotation = f'n = {aggregated_baselined["X"].shape[0]} trials'
        pupil_plot[1].annotate(annotation, xy=(0.3, 1.02), xycoords=pupil_plot[1].get_xaxis_transform())
        pupil_plot[1].axvspan(0, 0.15, color='grey', alpha=0.1)
        pupil_plot[1].axvspan(0.5, 0.65, color='grey', alpha=0.1)
        pupil_plot[1].axvspan(1, 1.15, color='grey', alpha=0.1)
        pupil_plot[1].axvspan(1.5, 1.65, color='grey', alpha=0.1)
        pupil_plot[1].set_ylabel('Pupil size (a.u.)')
        pupil_plot[1].set_xlabel('Time from stimulus onset (s)')
        #pupil_plot[1].set_ylim(Y_LIMS.get(animals_to_list, (-0.5,0.5)))
        #pupil_plot[0].suptitle(f'Baseline Regressed plot for {animals_to_list}')
        pupil_plot[0].suptitle(f'Baseline Regressed plot for {animals_to_list}')
        fig = plt.gcf()
        if show_plot:
            pupil_plot[0].show()
        if save_figure:
            fig.savefig(self.output_path / fr'Stage {self.stage} Baseline Regression\{animals_to_list}_Baseline_Regressed.svg')
        plt.pause(0.1)
        fig.clf()