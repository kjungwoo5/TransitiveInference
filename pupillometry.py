import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import os

import sys
sys.path.append('../')
sys.path.append(str(Path(__file__).parent.parent.parent))
from Analysis.XdetectionCore.xdetectioncore.plotting import plot_shaded_error_ts, format_axis

READING_WINDOW = [-2, 5]
PLOTTING_WINDOW = [-1, 4]

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
    'A': 'oldlace', 
    'B': 'blanchedalmond', 
    'C': 'wheat', 
    'D': 'burlywood', 
    'E': 'sandybrown', 
    'F': 'peru', 
    'G': 'sienna', 
    'H': 'saddlebrown', 
    'I': 'firebrick', 
    'J': 'darkred',
    'CFED': 'r',
}

OUTPUT_SUBDIRS = {
    'testing': 'Testing Phase',
    'exposure': 'Exposure Phase',
    'first': 'First Patterns Only',
    'second': 'Second Patterns Only',
}

ALL_ANIMALS = ['JK01', 'JK02', 'JK03', 'JK04']

Y_LIMS = {
    'JK01': (-0.25,0.35),
    'JK02': (-0.5,0.7),
    'JK03': (-0.5,0.4),
    'JK04': (-0.5,0.6),
    ' JK01, JK02, JK03, JK04 ': (-0.35,0.35)
}

class PupilPlotter:
    def __init__(self, pupil_df: pd.DataFrame, harp_df: pd.DataFrame, stage: int, type_of_analysis: str, output_path: Path, animals: list):
        valid_types_of_analysis = {'testing', 'exposure', 'first', 'second'}
        if type_of_analysis not in valid_types_of_analysis:
            raise Exception('Not a valid type of analysis! (\'testing\', \'exposure\', \'first\', \'second\')')
        if type_of_analysis in {'first', 'second'} and stage < 3:
            raise Exception(f'{type_of_analysis} is not valid for stage {stage}!')
        

        self.stage = stage
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
        
        
    def get_stimuli(self, harp):
        if self.type_of_analysis != 'exposure':
            # Take harp data only past the first 100 trials (i.e. occurrences of X)
            Xs = harp.index[harp['Payload'] == 3].tolist()
            if len(Xs) > 100:
                harp = harp[harp.index >= Xs[100]]
                Xs = Xs[100:]

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
        
        elif self.type_of_analysis == 'exposure':
            # Take harp data only until the first 100 trials (i.e. occurrences of X)
            Xs = harp.index[harp['Payload'] == 3].tolist()
            if len(Xs) > 100:
                Xs = Xs[:100]
                harp = harp[harp.index < Xs[100]]

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
    def align_pupil_by_session(self):
        session_ids = self.harp_df['session_id'].unique()
        aligned_pupil_by_session = {}
        for session_id in session_ids:
            pupil = self.pupil_df[self.pupil_df['session_id'] == session_id]['pupilsense_raddi_a_zscored']
            harp = self.harp_df[self.harp_df['session_id'] == session_id]

            stimuli_list = self.get_stimuli(harp)
            
            for stimulus in stimuli_list:
                for index in stimulus:
                    harp.at[index, 'id'] = self.types_of_stimuli[stimuli_list.index(stimulus)]
            
            event_times_by_event = {}
            aligned_pupil = {}
            for event_id in self.types_of_stimuli:
                event_times_by_event[event_id] = harp[harp['id']==event_id]['Timestamp'].values
                
            for event_id, event_times in event_times_by_event.items():
                if self.stage == 1:
                    epochs = [pupil.loc[t -0.5 :t + 1] for t in event_times]
                else:
                    epochs = [pupil.loc[t + READING_WINDOW[0]:t + READING_WINDOW[1]] for t in event_times]
                if epochs == []:
                    continue
                epochs_array = np.full((len(epochs),max([len(e) for e in epochs])), np.nan)
                for index, epoch in enumerate(epochs):
                    epochs_array[index][:len(epoch)] = epoch.values
                aligned_pupil[event_id] = epochs_array[-300:]

            if self.stage == 1:
                x_ser = np.round(np.linspace(-0.5, 1, aligned_pupil['X'].shape[1]), 2)
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
                fig.savefig(self.output_path / fr'Whole Session Pupils\{session}_fullsession.png')
            fig.clf()
            plt.close()
            

    def plot_pupil_by_session(self, save_figure = True, show_plot = True):
        valid_types_of_analysis = {'testing', 'exposure', 'first', 'second'}
        if self.type_of_analysis not in valid_types_of_analysis:
            raise Exception('Not a valid type of analysis! (\'testing\', \'exposure\', \'first\', \'second\')')

        for session, value in self.aligned_pupil_by_session.items():
            plt.pause(0.1)
            pupil_plot = plt.subplots()
            print('Plotting pupil plot for session: ', session)

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
                pupil_plot[1].plot(response.columns, response.mean(axis=0), label=event_id, color=STIMULUS_COLOURS.get(event_id, None))
                plot_shaded_error_ts(pupil_plot[1], response.columns, response.mean(axis=0),
                                    response.sem(axis=0), alpha=0.1, color=STIMULUS_COLOURS.get(event_id, None))
            pupil_plot[1].legend()
            pupil_plot[1].set_xlim((PLOTTING_WINDOW[0], PLOTTING_WINDOW[1]))
            annotation = f'n = {total_responses["X"].shape[0]} trials'
            pupil_plot[1].annotate(annotation, xy=(0.3, 1.02), xycoords=pupil_plot[1].get_xaxis_transform())
            pupil_plot[1].axvspan(0, 0.15, color='grey' , alpha=0.1)
            pupil_plot[1].axvspan(0.5, 0.65, color='grey', alpha=0.1)
            pupil_plot[1].axvspan(1, 1.15, color='grey', alpha=0.1)
            pupil_plot[1].axvspan(1.5, 1.65, color='grey', alpha=0.1)
            #pupil_plot[1].set_ylim((-0.5,0.9))
            pupil_plot[0].suptitle(f'Nonbaselined plot for: {session}')
            fig = plt.gcf()
            if show_plot:
                pupil_plot[0].show()
            if save_figure:
                os.makedirs(fr'{self.output_path}\{self.output_subdir}\Individual Sessions', exist_ok=True)
                fig.savefig(
                    fr'{self.output_path}\{self.output_subdir}\Individual Sessions\Stage{self.stage}_{session}_nonbaselined.png')
            fig.clf()

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
                    fr'{self.output_path}\{self.output_subdir}\Individual Sessions\Stage{self.stage}_{session}.png')
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
                    fr'{self.output_path}\{self.output_subdir}\Actual Distributions\Stage{self.stage}_{session}_distribution.png'
                )
            fig.clf()

    def aggregate_total(self) -> dict:
        total_responses = {}
        for stimulus in self.types_of_stimuli:
            aggregate = []
            for key, value in self.aligned_pupil_by_session.items():
                if stimulus in self.aligned_pupil_by_session[key]:
                    aggregate.append(self.aligned_pupil_by_session[key][stimulus])
            total_responses[stimulus] = pd.concat(aggregate, axis=0, ignore_index=True)
        return total_responses

    def plot_overall_pupil(self, save_figure = True, show_plot = True):
        animals_to_list = ', '.join(self.animals)

        aggregated_aligned_pupil = self.aggregate_total()
        pupil_plot = plt.subplots()
        for event_id, response in aggregated_aligned_pupil.items():
            pupil_plot[1].plot(response.columns, response.mean(axis=0),label=event_id, color=STIMULUS_COLOURS.get(event_id, None))
            plot_shaded_error_ts(pupil_plot[1],response.columns,response.mean(axis=0), response.sem(axis=0),alpha=0.1, color=STIMULUS_COLOURS.get(event_id, None))
        pupil_plot[1].legend()
        pupil_plot[1].set_xlim((PLOTTING_WINDOW[0], PLOTTING_WINDOW[1]))
        annotation = f'n = {aggregated_aligned_pupil["X"].shape[0]} trials'
        pupil_plot[1].annotate(annotation, xy=(0.3, 1.02), xycoords=pupil_plot[1].get_xaxis_transform())
        pupil_plot[1].axvspan(0, 0.15, color='grey', alpha=0.1)
        pupil_plot[1].axvspan(0.5, 0.65, color='grey', alpha=0.1)
        pupil_plot[1].axvspan(1, 1.15, color='grey', alpha=0.1)
        pupil_plot[1].axvspan(1.5, 1.65, color='grey', alpha=0.1)
        pupil_plot[0].suptitle(f'Non-baselined plot for {animals_to_list}')
        fig = plt.gcf()
        if show_plot:
            pupil_plot[0].show()
        if save_figure:
            os.makedirs(fr'{self.output_path}\{self.output_subdir}\{animals_to_list}', exist_ok=True)
            fig.savefig(fr'{self.output_path}\{self.output_subdir}\{animals_to_list}\Stage{self.stage}_{animals_to_list}_nonbaselined.png')
        fig.clf()
    
    def plot_overall_baseline_sub_aligned_pupil(self, save_figure = True, show_plot = True):
        animals_to_list = ', '.join(self.animals)

        aggregated_aligned_pupil = self.aggregate_total()
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
        pupil_plot[0].suptitle(f'Baseline subtracted plot for {animals_to_list}')
        fig = plt.gcf()
        if show_plot:
            pupil_plot[0].show()
        if save_figure:
            os.makedirs(fr'{self.output_path}\{self.output_subdir}\{animals_to_list}', exist_ok=True)
            fig.savefig(fr'{self.output_path}\{self.output_subdir}\{animals_to_list}\Stage{self.stage}_{animals_to_list}_Baseline_Subtracted.png')
        fig.clf()
    
    def plot_baseline_sub_training(self, save_figure = True, show_plot = True):
        animals_to_list = ', '.join(self.animals)
        pupil_plot = plt.subplots()
        n_stimuli = 0
        aggregated_aligned_pupil = self.aggregate_total()
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
        if show_plot:
            pupil_plot[0].show()
        if save_figure:
            os.makedirs(fr'{self.output_path}\{self.output_subdir}\{animals_to_list}', exist_ok=True)
            fig.savefig(fr'{self.output_path}\{self.output_subdir}\{animals_to_list}\Stage{self.stage}_{animals_to_list}_Baseline_Subtracted_Training.png')
        plt.pause(0.1)

    def plot_baseline_sub_testing(self, save_figure = True, show_plot = True):
        animals_to_list = ', '.join(self.animals)
        pupil_plot = plt.subplots()
        n_stimuli = 0
        aggregated_aligned_pupil = self.aggregate_total()
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
        pupil_plot[1].set_ylim(Y_LIMS.get(str(self.animals), (-0.5,0.5)))
        pupil_plot[1].axvspan(0, 0.15, color='grey', alpha=0.1)
        pupil_plot[1].axvspan(0.5, 0.65, color='grey', alpha=0.1)
        pupil_plot[1].axvspan(1, 1.15, color='grey', alpha=0.1)
        pupil_plot[1].axvspan(1.5, 1.65, color='grey', alpha=0.1)
        pupil_plot[0].suptitle(f'Baseline Subtracted plot of testing stimuli for {self.animals}')
        fig = plt.gcf()
        if show_plot:
            pupil_plot[0].show()
        if save_figure:
            os.makedirs(fr'{self.output_path}\{self.output_subdir}\{self.animals}', exist_ok=True)
            fig.savefig(fr'{self.output_path}\{self.output_subdir}\{self.animals}\Stage{self.stage}_{self.animals}_Baseline_Subtracted_Testing.png')
        fig.clf()
        

    def plot_overall_distribution(self, save_figure = True, show_plot = True):
        animals_to_list = ', '.join(self.animals)
        aggregated_aligned_pupil = self.aggregate_total()
        plt.pause(0.1)
        dist_plot = plt.subplots()
        actual_distribution = {}
        for stimulus in self.types_of_stimuli:
            actual_distribution[stimulus] = len(aggregated_aligned_pupil[stimulus])
            dist_plot[1].bar(stimulus, actual_distribution[stimulus], color=STIMULUS_COLOURS.get(stimulus, None))
            dist_plot[1].text(stimulus, actual_distribution[stimulus] + 5, str(actual_distribution[stimulus]), ha='center',
                            va='center')

        dist_plot[0].suptitle(f'Overall distribution for: {animals_to_list}')
        annotation = f'n = {aggregated_aligned_pupil["X"].shape[0]} trials'
        dist_plot[1].annotate(annotation, xy=(0.3, 1.02), xycoords=dist_plot[1].get_xaxis_transform())
        fig = plt.gcf()
        if show_plot:
            dist_plot[0].show()
        if save_figure:
            os.makedirs(fr'{self.output_path}\{self.output_subdir}\Actual Distributions', exist_ok=True)
            fig.savefig(fr'{self.output_path}\{self.output_subdir}\Actual Distributions\Stage{self.stage}_{animals_to_list}_distribution.png')
        fig.clf()
        