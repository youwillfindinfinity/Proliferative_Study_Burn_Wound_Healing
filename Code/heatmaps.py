import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

# Output folder of the pickle files
output_folder = "E:/Author release/Proliferative_Study_Burn_Wound_Healing_author_release/results_D"
scn = [1, 2]
for sc in scn:
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
    #
    percentages = ['', '25p', '50p', '75p'] # list of fragmented data
    varnames = ["A_MII", "A_MC", "I", "beta", "A_F", "A_M", "A_Malpha", "CI", "CIII"] # variables 
    data = []
    # Change sc accordingly
    for var in varnames:
        for i in range(len(percentages)):
            with open(output_folder + '/' + 'sobol_indices_' + f'sc{sc}' + '_' + str(var) + '.pickle','rb') as f:
                k = pickle.load(f)
                data.append(k['ST'])
                

    data = pd.DataFrame(data, columns = columns)

    filename1 = f"S{sc}_all_vars{sc}.pickle"
    # filename2 = "S2_all_vars2.pickle"

    # Save the data to the pickle file, change file name accordingly
    with open(output_folder+"/" + filename1, 'wb') as f:
        pickle.dump(data, f)
    # Load the data to the pickle file, change file name accordingly
    with open(output_folder + '/' + filename1,'rb') as f:
        k = pickle.load(f)

    columns_labels = [r'$k_1$', r'$k_2$', r'$k_3$', r'$k_4$', r'$k_5$', r'$k_6$', r'$k_7$', r'$k_8$', r'$k_9$', r'$k_{10}$',
    r'$\gamma$', r'$\zeta$',
    r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$', r'$\lambda_4$',
    r'$\rho_1$', r'$\rho_2$', r'$\rho_3$',
    r'$\mu_1$', r'$\mu_2$', r'$\mu_3$', r'$\mu_4$', r'$\mu_5$', r'$\mu_6$', r'$\mu_7$', r'$\mu_8$',
    r'$\upsilon_1$', r'$\upsilon_2$', r'$\upsilon_3$', r'$\upsilon_4$',
    r'$\omega_1$', r'$\omega_2$', r'$\omega_3$',
    r'$A_{MII0}$', r'$I_0$', r'$T_0$', r'$A_{MC0}$', r'$A_{F0}$', r'$A_{M0}$', r'$A_{M\alpha0}$',
    r'$C_{III0}$', r'$C_{I0}$'
    ]
    # Create a heatmap using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(k, annot=False, cmap='GnBu')
    # Set custom x-axis labels
    if sc == 1:
        letter = "a"
    if sc == 2:
        letter = "b"
    plt.xticks(ticks=np.arange(len(columns_labels)) + 0.5, labels=columns_labels, rotation=80)
    plt.title(f'({letter}) - Sobol Indices Heatmap for Scenario {sc}') # Change title accordingly
    plt.xlabel('Parameters')
    plt.ylabel('Samples')
    plt.savefig(f"SA_heatmap_sc{sc}.png", dpi = 400) # Change file name accordigly
    plt.show()