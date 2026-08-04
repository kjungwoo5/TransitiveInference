import matplotlib as mpl
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path
from numpy.linalg import norm
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy import signal
from sklearn.linear_model import LinearRegression

import os

import sys
sys.path.append('../')
sys.path.append(str(Path(__file__).parent.parent.parent))
from Analysis.XdetectionCore.xdetectioncore.plotting import plot_shaded_error_ts, format_axis
from pupillometry import STIMULUS_COLOURS, PLOTTING_WINDOW, STAGE1_FREQUENCIES, Y_LIMS, PERMS_Y_LIMS


# STAGE 1
def plot_pitch_dependency(self, offset = 0.525, window_size = 0.25, save_figure = True, show_plot = True, omit_outliers = True, show_means_as_points = True, plot_quad = True):
    animal = [animal.split('_')[0] for animal in self.animals]
    animals_to_list = ', '.join(animal)
    
    if animals_to_list.strip() == 'JK01, JK02, JK03, JK04':
        animals_to_list = 'all mice'

    aggregated_aligned_pupil = self.aggregate_total()
    fig, ax = plt.subplots()
    plot_data = []
    labels = []

    def _filter_outliers(values):
        values = np.asarray(values, dtype=float)
        values = values[~np.isnan(values)]
        if values.size < 4:
            return values

        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return values[(values >= lower) & (values <= upper)]

    for event_id, response in aggregated_aligned_pupil.items():
        if event_id == 'X':
            continue
        baseline_mean = response.loc[:, -0.25:0.25].mean(axis=1)
        baselined = response.sub(baseline_mean, axis=0)
        window_slice = baselined.loc[:, round(offset - window_size / 2, 2):round(offset + window_size / 2, 2)]
        window_means = window_slice.mean(axis=1)
        filtered_values = _filter_outliers(window_means) if omit_outliers else window_means.to_numpy()
        if filtered_values.size > 0:
            plot_data.append(filtered_values)
            labels.append(event_id)

    if not plot_data:
        return


    means = [np.mean(values) for values in plot_data]
    
    #calculate equation for quadratic trendline
    i = np.arange(0,len(labels))
    z = np.polyfit(i, means, 2)
    p_quad = np.poly1d(z)

    #add trendline to plot
    if plot_quad == True:
        plt.plot(
            i,
            p_quad(i),
            color='gray', linestyle='--', label=f'Quadratic best fit line'
        )
        plt.legend()
    
    for label, mean in zip(labels, means):
        ax.scatter(label, mean, color='black', s=40)
        # ax.text(label, mean + 0.01, f'{mean:.2f}', ha='center', va='bottom', fontsize=8)


    
    ax.set_xlabel('Stimulus Frequency (Hz)')
    ax.set_xticklabels(STAGE1_FREQUENCIES.values())
    ax.set_ylabel(r'Pupil size (a.u.)')
    ax.margins(y=0.05)
    ax.set_title(f'Pitch dependency for {animals_to_list} \n')
    annotation = f'n = {aggregated_aligned_pupil["X"].shape[0]} trials'
    ax.annotate(annotation, xy=(0.005, 1.02), xycoords=ax.get_xaxis_transform())
    
    fig = plt.gcf()
    
    if show_plot:
        fig.show()
    if save_figure:
        os.makedirs(fr'{self.output_path}\Pitch Dependency', exist_ok=True)
        fig.savefig(fr'{self.output_path}\Pitch Dependency\{animals_to_list}_Pitch_Dependency_{offset}_{window_size}_{plot_quad}.svg')
    fig.clf()

def plot_stage1_window_diagram(self, stim_to_show: str, window = (0, 0.75), save_figure = True, show_plot = True): 
    animals_to_list = ', '.join(self.animals)
    animals_to_list = animals_to_list.split('_')[0]
    
    aggregated_aligned_pupil = self.aggregate_total()
    pupil_plot = plt.subplots()
    for event_id, response in aggregated_aligned_pupil.items():
        if event_id != stim_to_show:
            continue
        baseline_mean = response.loc[:, -0.05:0.05].mean(axis=1)
        baselined = response.sub(baseline_mean, axis=0)
        pupil_plot[1].plot(baselined.columns, baselined.median(axis=0),label=event_id, color=STIMULUS_COLOURS.get(event_id, None))
        plot_shaded_error_ts(pupil_plot[1],baselined.columns,baselined.median(axis=0), baselined.sem(axis=0),alpha=0.1, color=STIMULUS_COLOURS.get(event_id, None))
    pupil_plot[1].set_xlim((window[0], window[1]))
    annotation = f'n = {aggregated_aligned_pupil[stim_to_show].shape[0]} stimuli'
    pupil_plot[1].annotate(annotation, xy=(0.005, 1.02), xycoords=pupil_plot[1].get_xaxis_transform())
    pupil_plot[1].set_ylim((-0.05, 0.16))
    pupil_plot[1].set_ylabel(r'Pupil size (a.u.)')
    pupil_plot[1].set_xlabel('Time from stimulus onset (s)')
    
    pupil_plot[1].axvspan(0, 0.15, color='grey', alpha=0.1)
 
    
    bl = (0.45, -0.03)
    tr = (0.55, pupil_plot[1].get_ylim()[1] * 0.90)
    
    pupil_plot[1].plot([bl[0],tr[0]], [bl[1],bl[1]], '--', color= 'gray')
    pupil_plot[1].plot([tr[0],tr[0]], [bl[1], tr[1]], '--', color= 'gray')
    pupil_plot[1].plot([tr[0],bl[0]], [tr[1], tr[1]], '--', color= 'gray')
    pupil_plot[1].plot([bl[0],bl[0]], [tr[1],bl[1]], '--', color= 'gray')
    
    pupil_plot[0].suptitle(f'{animals_to_list} pupil response to {STAGE1_FREQUENCIES.get(stim_to_show)} Hz tone')
    fig = plt.gcf()
    if show_plot:
        pupil_plot[0].show()
    if save_figure:
        os.makedirs(fr'{self.output_path}\{self.output_subdir}\{animals_to_list}', exist_ok=True) 
        fig.savefig(fr'{self.output_path}\Stage 1\{animals_to_list}_{stim_to_show}_methods.svg')
    fig.clf()


def plot_stage1_peak_time_by_stimulus(self, baseline_window=(-0.15, 0.15), search_window=(0.15, 0.75), save_figure=True, show_plot=True):
    animals_to_list = ', '.join(self.animals)
    animals_to_list = animals_to_list.split('_')[0]

    aggregated_aligned_pupil = self.aggregate_total()
    stimuli = []
    peak_times = []
    stimulus_labels = []

    for event_id, response in aggregated_aligned_pupil.items():
        if event_id == 'X':
            continue

        baseline_mean = response.loc[:, baseline_window[0]:baseline_window[1]].mean(axis=1)
        baselined = response.sub(baseline_mean, axis=0)
        mean_trace = baselined.mean(axis=0)

        times = np.asarray(mean_trace.index, dtype=float)
        values = np.asarray(mean_trace.to_numpy(), dtype=float)

        start_idx = np.searchsorted(times, search_window[0], side='left')
        end_idx = np.searchsorted(times, search_window[1], side='right')

        if start_idx >= len(times):
            continue
        if end_idx <= start_idx:
            continue

        search_values = values[start_idx:end_idx]
        search_times = times[start_idx:end_idx]

        if search_values.size == 0:
            continue

        derivative = np.gradient(search_values, search_times)
        local_max_idx = None
        for idx in range(1, len(derivative) - 1):
            if derivative[idx - 1] > 0 and derivative[idx] <= 0:
                local_max_idx = idx
                break

        if local_max_idx is None:
            local_max_idx = int(np.argmax(search_values))

        peak_time = float(search_times[local_max_idx])
        stimuli.append(event_id)
        peak_times.append(peak_time)
        stimulus_labels.append(STAGE1_FREQUENCIES.get(event_id, event_id))

    if not stimuli:
        return

    fig, ax = plt.subplots()
    col = ['grey' if peak_time in {0.15, 0.75} else 'black' for peak_time in peak_times]
    ax.scatter(stimulus_labels, peak_times, color=col, s=50)
    for x, y, label, color in zip(stimulus_labels, peak_times, stimuli, col):
        ax.annotate(y, (x, y), textcoords='offset points', xytext=(0, 6), ha='center', fontsize=8, color=color)
    peak_times_inliers = [peak_time for peak_time in peak_times if peak_time != 0.15 and peak_time != 0.75]
    print(f"{animals_to_list}: Mean peak time: {np.mean(peak_times_inliers):.3f} s, Std Dev: {np.std(peak_times_inliers):.3f} s")
    ax.set_xlabel('Stimulus frequency (Hz)')
    ax.set_ylabel('Time to peak (s)')
    ax.set_ylim(0.1,0.8)
    ax.set_title(f'Time from stimulus onset to peak pupil size for {animals_to_list}')
    ax.grid(True, alpha=0.3)

    if show_plot:
        fig.show()
    if save_figure:
        os.makedirs(fr'{self.output_path}\Stage 1', exist_ok=True)
        fig.savefig(fr'{self.output_path}\Stage 1\{animals_to_list}_local_max_time_by_stimulus.svg')

    fig.clf()
    return np.mean(peak_times_inliers)


# STAGE 4


    

def plot_difference(self, normal_stim: str, deviant_stim: str, window: tuple, by_session = True, show_plot = True, save_figure = True, regress_baseline = False):
    animals_to_list = ', '.join(self.animals)
    
    
    baseline_window = (-1,0)
    df_for_regr = self.build_session_wide_baseline_dataframe(
        self.aligned_pupil_by_session,
        include_events=[event_id for event_id in self.types_of_stimuli if event_id != 'X'],
        smooth_window=9
    )
    coefs, r2, r2_by_feature = self.fit_time_resolved_baseline_regression(df_for_regr, base_window=baseline_window)

    # Apply the session-wide baseline regression back to every session.
    baselined_sessions = self.regress_out_baseline_across_sessions(
        self.aligned_pupil_by_session,
        coefs[['immediate','slope','mean']],
        base_window=baseline_window,
        events_to_regress=[event_id for event_id in self.types_of_stimuli if event_id != 'X']
    )
    
    pupil_plot = plt.subplots()
    
    aligned_pupil_by_session = self.aligned_pupil_by_session
    if regress_baseline: 
        aligned_pupil_by_session = baselined_sessions
    
    if by_session: 
        observed_differences = []
        shuffled_differences = []
        for session in self.pupil_df['session_id'].unique():
            if len(aligned_pupil_by_session[session].keys()) < 5:
                continue
            baseline_dev = aligned_pupil_by_session[session][deviant_stim].loc[:, -0.25:0.25].mean(axis=1)
            baseline_norm = aligned_pupil_by_session[session][normal_stim].loc[:, -0.25:0.25].mean(axis=1)
            baselined_dev = aligned_pupil_by_session[session][deviant_stim].sub(baseline_dev, axis=0)
            baselined_norm = aligned_pupil_by_session[session][normal_stim].sub(baseline_norm, axis=0)
            
            dev_pupil = pd.DataFrame(baselined_dev.loc[:, window[0]:window[1]].mean(axis=1))
            dev_pupil['id'] = deviant_stim
            
            norm_pupil = pd.DataFrame(baselined_norm.loc[:, window[0]:window[1]].mean(axis=1))
            norm_pupil['id'] = normal_stim

            pupils = pd.concat([dev_pupil, norm_pupil], axis = 0).reset_index()
            shuffled_pupils = pupils.copy()
            
            rng = np.random.default_rng(3)
            shuffled_pupils['id'] = rng.permutation(shuffled_pupils['id'])
            
            observed_difference = pupils.groupby(['id'])[0].mean()[deviant_stim] - \
                                pupils.groupby(['id'])[0].mean()[normal_stim]
            
            shuffled_difference = shuffled_pupils.groupby(['id'])[0].mean()[deviant_stim] - \
                                shuffled_pupils.groupby(['id'])[0].mean()[normal_stim]
            
            observed_differences.append(observed_difference)
            shuffled_differences.append(shuffled_difference)
            
            
        data = [observed_differences, shuffled_differences]
        
        wilcoxon = stats.wilcoxon(data[0], data[1], alternative='greater')
        
        box = pupil_plot[1].boxplot(data, labels=["Data", "Shuffle"], patch_artist=True)
        for patch in box["boxes"]:
            patch.set(facecolor="lightgray", alpha=0.8)
        
        pupil_plot[0].suptitle(f'{animals_to_list} difference in pupil dilation for deviant')
        pupil_plot[0].text(
                0.98,
                0.95,
                f"p = {wilcoxon.pvalue:.3f}",
                transform=plt.gca().transAxes,
                ha="right",
                va="top",
                fontsize=10,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )
        
        
        fig = plt.gcf()
        fig.savefig(fr'{self.output_path}\Differences\{window}_{regress_baseline}_{animals_to_list}_Differences.svg')
        fig.clf()

def plot_differences_method(self, normal_stim: str, deviant_stim: str, window: tuple, show_plot = True, save_figure = True):
    animals_to_list = ', '.join(self.animals)
    animals_to_list = animals_to_list.split('_')[0]
    
    aggregated_aligned_pupil = self.aggregate_total()
    pupil_plot = plt.subplots()
    for event_id, response in aggregated_aligned_pupil.items():
        if event_id != normal_stim and event_id != deviant_stim:
            continue
        baseline_mean = response.loc[:, -1:0.15].mean(axis=1)
        baselined = response.sub(baseline_mean, axis=0)
        pupil_plot[1].plot(baselined.columns, baselined.mean(axis=0),label=event_id, color=STIMULUS_COLOURS.get(event_id, None))
        plot_shaded_error_ts(pupil_plot[1],baselined.columns,baselined.mean(axis=0), baselined.sem(axis=0),alpha=0.1, color=STIMULUS_COLOURS.get(event_id, None))
    pupil_plot[1].set_xlim((PLOTTING_WINDOW[0], PLOTTING_WINDOW[1]))
    annotation = f'n {normal_stim} = {aggregated_aligned_pupil[normal_stim].shape[0]}, n {deviant_stim} = {aggregated_aligned_pupil[deviant_stim].shape[0]}'
    pupil_plot[1].annotate(annotation, xy=(-1.005, 1.02), xycoords=pupil_plot[1].get_xaxis_transform())
    pupil_plot[1].set_ylim(ymax= pupil_plot[1].get_ylim()[1] * 1.05)
    pupil_plot[1].set_ylabel(r'Pupil size (a.u.)')
    pupil_plot[1].set_xlabel('Time from stimulus onset (s)')
    pupil_plot[1].legend(loc = 'upper left')
    pupil_plot[1].axvspan(0, 0.15, color='grey', alpha=0.1)
    pupil_plot[1].axvspan(0.5, 0.65, color='grey', alpha=0.1)
    pupil_plot[1].axvspan(1, 1.15, color='grey', alpha=0.1)
    pupil_plot[1].axvspan(1.5, 1.65, color='grey', alpha=0.1)
    
    
    bl = (window[0], pupil_plot[1].get_ylim()[0] * 0.90)
    tr = (window[1], pupil_plot[1].get_ylim()[1] * 0.90)
    
    pupil_plot[1].plot([bl[0],tr[0]], [bl[1],bl[1]], '--', color= 'gray')
    pupil_plot[1].plot([tr[0],tr[0]], [bl[1], tr[1]], '--', color= 'gray')
    pupil_plot[1].plot([tr[0],bl[0]], [tr[1], tr[1]], '--', color= 'gray')
    pupil_plot[1].plot([bl[0],bl[0]], [tr[1],bl[1]], '--', color= 'gray')
    
    pupil_plot[0].suptitle(f'Window used to calculate Δ pupil size for testing stimuli')
    
    # pupil_plot[1].spines[['right', 'top']].set_visible(False)
    
    fig = plt.gcf()
    if show_plot:
        pupil_plot[0].show()
    if save_figure:
        os.makedirs(fr'{self.output_path}\{self.output_subdir}\{animals_to_list}', exist_ok=True) 
        fig.savefig(fr'{self.output_path}\Stage 4\{animals_to_list}_differences_methods.svg')
    fig.clf()

def plot_cosine_similarity(self, stim1id: str, stim2id: str, window: tuple, show_plot = True, save_figure = True):
    
    animals_to_list = ', '.join(self.animals)
    pupil_plot = plt.subplots()
    print(f'Cosine similarities for {self.animals}')
    cosine_similarities = []
    sessions = []
    for session in self.pupil_df['session_id'].unique():
        if len(self.aligned_pupil_by_session[session].keys()) < 5:
            continue
        sessions.append(session)
        baseline_1 = self.aligned_pupil_by_session[session][stim1id].loc[:, -1:0].mean(axis=1)
        baseline_2 = self.aligned_pupil_by_session[session][stim2id].loc[:, -1:0].mean(axis=1)
        baselined_1 = self.aligned_pupil_by_session[session][stim1id].sub(baseline_1, axis=0)
        baselined_2 = self.aligned_pupil_by_session[session][stim2id].sub(baseline_2, axis=0)
        
        stim1 = baselined_1.loc[:, window[0]:window[1]].mean(axis=0)
        
        stim2 = baselined_2.loc[:, window[0]:window[1]].mean(axis=0)

        cosine_similarity = np.dot(stim1, stim2) / (norm(stim1) * norm(stim2))
        print(f'Cosine similarity for {stim1id} and {stim2id} is {cosine_similarity}.')
        cosine_similarities.append(cosine_similarity)
        
    # pupil_plot[1].scatter(sessions, cosine_similarities)
    session_numbers = np.arange(1, len(sessions) + 1)
    coef = np.polyfit(session_numbers, cosine_similarities, 1)
    poly1d_fn = np.poly1d(coef)
    pupil_plot[1].scatter(session_numbers, cosine_similarities)
    
    model = LinearRegression()
    model.fit(session_numbers.reshape(-1,1), cosine_similarities)
    
    r2 = model.score(session_numbers.reshape(-1,1), cosine_similarities)
    coefs = model.coef_
    intercept = model.intercept_
    
    r_squared = r'$R^2 = $' + str(round(r2,2))
    
    pupil_plot[1].plot(session_numbers, poly1d_fn(session_numbers), '--r', label = r_squared)
    
    pupil_plot[0].suptitle(f'{animals_to_list} cosine similarities in pupil dilation for training stimuli')
    pupil_plot[1].legend(loc = 'top left')
        
    
    fig = plt.gcf()
    fig.savefig(fr'{self.output_path}\Similarities\Stage{self.stage}_{animals_to_list}_Similarities.svg')
    fig.clf()


# STAGE 5
def plot_pupil_preprocessing(self, session_id: str = "260625_000", show_plot = True, save_figure = True):
    if len(self.animals) > 1:
        raise ValueError("Plotting pupil preprocessing only works for one animal at a time.")
    
    animal = ''.join(self.animals).strip('_filtered')
    
    y_labels = [
        'Processed pupil data',
        'Raw pupil data', 
    ]
    
    session = '_'.join((animal, session_id))
    filtered_pupil = self.pupil_df[self.pupil_df['session_id'] == session]
    filtered_pupil = filtered_pupil.reset_index()
    filtered_pupil['Time (min)'] = (filtered_pupil['index'] - filtered_pupil['index'][0]) / 60
    
    raw_pupil = pd.read_csv(fr"X:\Dammy\mouse_pupillometry\mouse_hf\{animal}_{session_id}\{animal}_{session_id.split('_')[0]}_eye0_eye_ellipse.csv", header = 0)
    raw_pupil['Time (min)'] = raw_pupil['frame_num'] / (60 * 90)
    
    fig, ax = plt.subplots(figsize = (16, 4))
    axes = [ax, ax.twinx()]
    colours = ('tab:orange', 'tab:grey')
    
    pupils = [filtered_pupil, raw_pupil]
    sizes = ['pupilsense_raddi_a_zscored', 'radius']
    
    for i, (ax, colour) in enumerate(zip(axes, colours)):
        ax.plot(pupils[i]['Time (min)'], pupils[i][sizes[i]], color = colour, alpha=0.7)
        ax.set_ylabel(y_labels[i], color = colour, fontsize = 'large')
        ax.spines.top.set_visible(False)
    axes[0].set_xlabel('Time (min)', fontsize = 'large')
    
    plt.savefig(fr'{self.output_path}\Raw Pupils\{session}_pupil_preprocessing.svg')
    fig.clear()
    
    
    fig, ax = plt.subplots(figsize = (16, 4))
    axes = [ax, ax.twinx()]
    colours = ('tab:orange', 'tab:grey')
    
    filtered_pupil = filtered_pupil[filtered_pupil['Time (min)'] <= 1]
    raw_pupil = raw_pupil[raw_pupil['Time (min)'] <= 1]
    
    filtered_pupil['Time (s)'] = filtered_pupil['Time (min)'] * 60
    raw_pupil['Time (s)'] = raw_pupil['Time (min)'] * 60
    
    pupils = [filtered_pupil, raw_pupil]
    sizes = ['pupilsense_raddi_a_zscored', 'radius']
    
    for i, (ax, colour) in enumerate(zip(axes, colours)):
        ax.plot(pupils[i]['Time (s)'], pupils[i][sizes[i]], color = colour, alpha=0.7)
        ax.set_ylabel(y_labels[i], color = colour, fontsize = 'large')
        ax.spines.top.set_visible(False)
    axes[0].set_xlabel('Time (s)', fontsize = 'large')
    
    plt.savefig(fr'{self.output_path}\Raw Pupils\{session}_pupil_preprocessing_sample.svg')
    fig.clear()

def prep_for_decoding(self, window_size = 0.1, tmax = 0.64, time_from_next_pip = None):
    pip_dilations = []
    aggregated_aligned_pupil = self.aggregate_total(baseline_data=False)
    
    # Correct each individual trial such that baseline is around 0 at 0s
    for event_id, response in aggregated_aligned_pupil.items():
        pip_dilation = []
        for index, row in response.iterrows():
            baseline_mean = row.loc[-0.15:0.15].mean()
            row = row.sub(baseline_mean)
            
            if time_from_next_pip == None: 
                # Get mean pupil dilation from windows of 100ms centred around 
                # tmax (640ms), tmax + 500ms, tmax + 1000ms, and tmax + 1500ms respectively
                a = row.loc[tmax-(window_size/2) : tmax+(window_size/2)].mean()
                b = row.loc[tmax+0.5-(window_size/2) : tmax+0.5+(window_size/2)].mean()
                c = row.loc[tmax+1.0-(window_size/2) : tmax+1.0+(window_size/2)].mean()
                d = row.loc[tmax+1.5-(window_size/2) : tmax+1.5+(window_size/2)].mean()
            else: 
                # Get mean pupil dilation from times going back from next pip offset time
                # 0.65-window, 1.15-window, 1.65-window, 2.15-window respectively
                a = row.loc[0.65-time_from_next_pip : 0.65].mean()
                b = row.loc[1.15-time_from_next_pip : 1.15].mean()
                c = row.loc[1.65-time_from_next_pip : 1.65].mean()
                d = row.loc[2.15-time_from_next_pip : 2.15].mean()
            pip_dilation.append([a,b,c,d, event_id])
        pip_dilations.extend(pip_dilation)
    # Convert pip_dilations into a df with columns of pip1, pip2, pip3, pip4, with label of event_id
    pip_df = pd.DataFrame(pip_dilations, columns=['pip1', 'pip2', 'pip3', 'pip4', 'stimulus_id'])
    return pip_df

def plot_stage5_perms(self, save_figure = True, show_plot = True):
    animals_to_list = ', '.join(self.animals)

    aggregated_aligned_pupil = self.aggregate_total()
    
    # Plot overall
    self.plot_overall_baseline_sub_aligned_pupil(save_figure=save_figure, show_plot=show_plot)
    
    # Plot by start letter
    for start_letter in 'CDEFA':
        pupil_plot = plt.subplots()
        n_stimuli = 0
        for event_id, response in aggregated_aligned_pupil.items():
            if event_id == 'X':
                continue
            if (start_letter != 'A') and (event_id[0] != start_letter):
                continue
            elif (start_letter == 'A') and ((event_id[0] != start_letter) and (event_id[0] != 'G')): 
                continue
            n_stimuli += len(response.index)
            baseline_mean = response.loc[:, -0.15:0.15].mean(axis=1)
            baselined = response.sub(baseline_mean, axis=0)
            pupil_plot[1].plot(baselined.columns, baselined.mean(axis=0),label=event_id, color=STIMULUS_COLOURS.get(event_id, None))
            plot_shaded_error_ts(pupil_plot[1],baselined.columns,baselined.mean(axis=0), baselined.sem(axis=0),alpha=0.1, color=STIMULUS_COLOURS.get(event_id, None))
        pupil_plot[0].suptitle(f'{animals_to_list.strip("_filtered")} pupil response to {start_letter}-starting sequences')
        pupil_plot[1].legend(loc = 'upper left')
        pupil_plot[1].set_xlim((PLOTTING_WINDOW[0], PLOTTING_WINDOW[1]))
        annotation = f'n = {n_stimuli} stimuli'
        pupil_plot[1].annotate(annotation, xy=(0.3, 1.02), xycoords=pupil_plot[1].get_xaxis_transform())
        pupil_plot[1].set_ylim(PERMS_Y_LIMS.get(animals_to_list, None))
        pupil_plot[1].set_ylabel('Pupil size (a.u.)')
        pupil_plot[1].set_xlabel('Time from stimulus onset (s)')
        pupil_plot[1].axvspan(0, 0.15, color='grey', alpha=0.1)
        pupil_plot[1].axvspan(0.5, 0.65, color='grey', alpha=0.1)
        pupil_plot[1].axvspan(1, 1.15, color='grey', alpha=0.1)
        pupil_plot[1].axvspan(1.5, 1.65, color='grey', alpha=0.1)
        fig = plt.gcf()
        if show_plot:
            pupil_plot[0].show()
        if save_figure:
            os.makedirs(fr'{self.output_path}\{self.output_subdir}\{animals_to_list}', exist_ok=True)
            fig.savefig(fr'{self.output_path}\{self.output_subdir}\{animals_to_list}\Stage{self.stage}_{start_letter}Sequences_{animals_to_list}_Baseline_Subtracted.svg')
        fig.clf()
        