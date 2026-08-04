import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path


OUTPUT_PATH = Path(r"C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\Outputs")


# data = pd.read_csv(r'C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\permutation_test_results.csv', header = 0, index_col=0)
data = pd.read_json(r'C:\Users\kjung\Documents\UCL\Year 4\ANAT0021 Dissertation\Coding\Analysis\Outputs\Permutation Tests\permutation_test_results_999.json')

types_of_decoding = ['first_tone', 'second_tone', 'third_tone', 'fourth_tone', 'sequence_identity', 'C_sequence_identity', 'D_sequence_identity', 'E_sequence_identity', 'F_sequence_identity']
animals = ['JK01', 'JK02', 'JK03', 'JK04']

m = len(types_of_decoding) * len(animals)

holm_thresholds_one = []
holm_thresholds_two = []

p_values = []

one_star_sig = 0.05
two_star_sig = 0.01

for type_of_decoding in types_of_decoding: 
    for animal in animals: 
        p_values.append([animal, type_of_decoding, data.loc[type_of_decoding, animal]['observed_accuracy'], data.loc[type_of_decoding, animal]['p_value']])

p_values.sort(key= lambda x: x[3])
print(p_values)

for i, p_value in enumerate(p_values):
    
    p_values[i].append(i)
    
    holm_threshold_one = one_star_sig / (m - i)
    holm_threshold_two = two_star_sig / (m - i)
    print(i, holm_threshold_one)
    
    holm_thresholds_one.append(holm_threshold_one)
    holm_thresholds_two.append(holm_threshold_two)
    
    if p_value[3] < holm_threshold_two:
        p_values[i].append('**')
    elif p_value[3] < holm_threshold_one:
        p_values[i].append('*')
    else:
        p_values[i].append('ns')


# print(p_values)

df = pd.DataFrame(p_values, columns = ['animal', 'decoding_type', 'decoded_accuracy', 'p_value', 'id', 'significance'])

print('Signficant decodings: \n', df[df['significance'] == '*'])

fig, ax = plt.subplots()

ax.scatter(df[df['significance'] == '*']['id'], df[df['significance'] == '*']['p_value'], c='r')
ax.scatter(df[df['significance'] == 'ns']['id'], df[df['significance'] == 'ns']['p_value'], c='k')
ax.plot([0,36], [0.05,0.05], linestyle='dashed', color = 'gray', label= 'Uncorrected threshold')
ax.plot(holm_thresholds_one, linestyle='dashed', color = 'r', label = 'Holm-Bonferroni threshold')
ax.plot([0,36], [0.05/36,0.05/36], linestyle='dashed', color = 'b', label= 'Bonferroni threshold')
ax.legend(loc='upper left')
ax.set_xlabel('Index of permutation tests sorted in ascending order of p-values')
ax.set_ylabel('p-values (log scale)')
ax.set_yscale('log')
ax.spines.top.set_visible(False)
ax.spines.right.set_visible(False)
ax.set_title('Visualisation of Holm-Bonferroni corrections \ncompared with other thresholds')

# plt.show()
plt.savefig(OUTPUT_PATH / 'Permutation Tests/Holm-Bonferroni corrected plot.svg')
