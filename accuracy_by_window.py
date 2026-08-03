from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.ticker as mtick

SESSION_PATH = Path(r"X:\Dammy\Xdetection_mouse_hf_test\session_topology_transitive_inference_cleaned.csv")
HOME_PATH = Path(r"C:\bonsai\data\JungWoo")
OUTPUT_PATH = Path(r"C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\Outputs")
# PARQUET_DIR = Path(r'X:\Dammy\mouse_pupillometry\pickles\trans_inf_test_90Hz_hpass00_lpass0')
PARQUET_DIR = Path(r'X:\Dammy\mouse_pupillometry\pickles\trans_inf_bandpass_90Hz_hpass01_lpass4')
HARP_DIR = Path(r'X:\Dammy\harpbins')

type_of_plot = {
    0: 'First position',
    1: 'Second position',
    2: 'Third position',
    3: 'Fourth position',
    4: 'Stimulus Identity',
    5: 'C-remaining sequence',
    6: 'D-remaining sequence',
    7: 'E-remaining sequence',
    8: 'F-remaining sequence',
}

for a in range(1,5):
    # Read data
    df = pd.read_json(fr'C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\bandpassed_accuracies_by_window_JK0{a}.json').transpose()
    print(df.head())

    for i, value in enumerate(type_of_plot.values()):
        #create scatterplot
        plt.scatter(df.index*1000, df[i])

        #calculate equation for quadratic trendline
        z = np.polyfit(df.index*1000, df[i], 2)
        p_quad = np.poly1d(z)
        print(p_quad)
        p = np.poly1d(z)

        #add trendline to plot

        equation = str(round(z[0],2)) + "x**2 + " + str(round(z[1],2)) + "x + " + str(round(z[2],2)) 
        plt.plot(
            df.index*1000,
            p_quad(df.index*1000),
            color='red', linestyle='--', label=f'Quadratic Trendline'
        )
        print(equation)
        plt.legend()
        plt.title(f'Decoding accuracy of {type_of_plot.get(i)} by window size for JK0{a}')
        plt.xlabel('Window size (ms)')
        plt.ylabel('Decoding accuracy')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1, decimals = 1))

        fig = plt.gcf()
        fig.savefig(fr'{OUTPUT_PATH}\Accuracies by window size\JK0{a}\Stage5_JK0{a}_{type_of_plot.get(i)}_accuracies.svg')
        fig.clear()
        
# Issue: very heterogeneous curves, have to decide what to base the decoding on. 