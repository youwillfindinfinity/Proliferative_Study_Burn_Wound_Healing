import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Define the folder containing the partitioned pickle files
output_folder = 'results_D'

columns = ['k_1', 'k_2', 'k_3', 'k_4', 'k_5', 'k_6', 'k_7', 'k_8', 'k_9', 'k_10',
'gamma', 'zeta',
'lambda_1', 'lambda_2', 'lambda_3', 'lambda_4',
'rho_1', 'rho_2', 'rho_3',
'mu_1', 'mu_2', 'mu_3', 'mu_4', 'mu_5', 'mu_6', 'mu_7', 'mu_8',
'upsilon_1', 'upsilon_2', 'upsilon_3', 'upsilon_4',
'omega_1', 'omega_2', 'omega_3',
'A_MII0', 'I_0', 'beta_0', 'A_MC0', 'A_F0', 'A_M0', 'A_Malpha0',
'C_III0', 'C_I0'
]

percentages = ['', '25p', '50p', '75p']
varnames = ["A_MII", "A_MC", "I", "beta", "A_F", "A_M", "A_Malpha", "CI", "CIII"]
ST = []
for var in varnames:
    for i in range(len(percentages)):
        with open(output_folder + '/' + 'sobol_indices_' + 'sc2' + '_' + str(var) + '.pickle','rb') as f:
            k = pickle.load(f)
            ST.append(k['ST'])
            # print(np.shape(k['ST']))

ST = pd.DataFrame(ST, columns = columns)
# print(ST)
filename = "S1_all_vars1.pickle"
# Save the data to the pickle file
with open(output_folder+"/" + filename, 'wb') as f:
    pickle.dump(ST, f)

with open(output_folder + '/' + filename,'rb') as f:
    k = pickle.load(f)
    # print(k)


# Create a heatmap using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(k, annot=False, cmap='Greens')
plt.title('Sobol Indices Heatmap for Scenario 1')
plt.xlabel('Parameters')
plt.ylabel('Samples')
plt.savefig("S1_sc1", dpi = 400)
plt.show()