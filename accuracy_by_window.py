import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


type_of_plot = {
    0: 'First position',
    1: 'Stimulus Identity'
}

for a in range(1,5):
    # Read data
    df = pd.read_json(fr'C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\accuracies_by_window_JK0{a}.json').transpose()
    print(df.head())

    for i in range(0,2):
        #create scatterplot
        plt.scatter(df.index, df[i])

        #calculate equation for quadratic trendline
        z = np.polyfit(df.index, df[i], 2)
        p_quad = np.poly1d(z)
        print(p_quad)
        p = np.poly1d(z)

        #add trendline to plot

        equation = str(round(z[0],2)) + "x**2 + " + str(round(z[1],2)) + "x + " + str(round(z[2],2)) 
        plt.plot(
            df.index,
            p_quad(df.index),
            color='red', linestyle='--', label=f'Quadratic Trendline: {equation}'
        )
        print(equation)
        plt.legend()
        plt.title(f'Decoding accuracy of {type_of_plot.get(i)} by window size for JK0{a}')
        plt.show()

        fig = plt.gcf()
        fig.clear()
        
# Issue: very heterogeneous curves, have to decide what to base the decoding on. 