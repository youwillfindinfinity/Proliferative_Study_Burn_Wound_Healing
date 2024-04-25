import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker

# Define your variables and load data as before
variables = ['A_F1', 'A_F2', 'A_M1', 'A_M2', 'A_Malpha1', 'A_Malpha2', 'A_MC1', 'A_MC2', 'A_MII1', 'A_MII2', 'beta1', 'beta2', 'CI1', 'CI2', 'CIII1', 'CIII2', 'I1', 'I2']
result_df_list = []

for var in variables:
    max_parameters_list = []
    min_parameters_list = []
    values_list = []
    labels = []

    paths = [
        "../Results/Data/best_init_params/best_initvalprob_{}.pkl".format(var), 
        "../Results/Data/best_init_params/worst_initvalprob_{}.pkl".format(var)
    ]

    for path in paths:
        with open(path, 'rb') as file:
            data = pickle.load(file)
            value = data['variable_value']
            parameters = data['initial_values']
            if path == paths[0]:
                labels.append('Max {}'.format(var))
                values_list.append(value)
                max_parameters_list.append(parameters)
            else:
                labels.append('Min {}'.format(var))
                values_list.append(value)
                min_parameters_list.append(parameters)

    # Create DataFrame for parameters
    parameters_df = pd.DataFrame(max_parameters_list + min_parameters_list)

    # Create DataFrame for values and labels
    values_df = pd.DataFrame({'Value': values_list, 'Label': labels})

    # Concatenate parameters and values DataFrames
    result_df = pd.concat([values_df, parameters_df], axis=1)
    result_df_list.append(result_df)

# Concatenate all result DataFrames vertically
final_result_df = pd.concat(result_df_list, ignore_index=True)

# Extract max and min values separately
max_values_df = final_result_df[final_result_df['Label'].str.startswith('Max')]
min_values_df = final_result_df[final_result_df['Label'].str.startswith('Min')]

# Drop 'Label' column as we don't want to plot it
max_values_df = max_values_df.drop(columns=['Label', 'Value'])
min_values_df = min_values_df.drop(columns=['Label', 'Value'])




with open("max_vals_df.pkl", "wb") as f:
    pickle.dump(max_values_df, f)

with open("min_vals_df.pkl", "wb") as f:
    pickle.dump(min_values_df, f)

# Compute feature importance (e.g., correlation with 'Value')
feature_importance_max = max_values_df.corrwith(final_result_df['Value'])
feature_importance_min = min_values_df.corrwith(final_result_df['Value'])


# Sort features by importance
feature_importance_max_sorted = feature_importance_max.abs().sort_values(ascending=False)
feature_importance_min_sorted = feature_importance_min.abs().sort_values(ascending=False)

# Define custom labels
variables_labels = [r'$A_{F}(Sc.1)$', r'$A_{F}(Sc.2)$', r'$A_{M}(Sc.1)$', r'$A_{M}(Sc.2)$', r'$A_{M\alpha}(Sc.1)$', r'$A_{M\alpha}(Sc.2)$', r'$A_{MC}(Sc.1)$', r'$A_{MC}(Sc.2)$', r'$A_{MII}(Sc.1)$', r'$A_{MII}(Sc.2)$', r'$T(Sc.1)$', r'$T(Sc.2)$', r'$C_{I}(Sc.1)$', r'$C_{I}(Sc.2)$', r'$C_{III}(Sc.1)$', r'$C_{III}(Sc.2)$', r'$I(Sc.1)$', r'$I(Sc.2)$']
xvariables_labelsmax = [r'$A_{MC0}$', r'$A_{MII0}$', r'$A_{F0}$', r'$A_{M0}$', r'$A_{M\alpha0}$', r'$T_0$', r'$I_0$', r'$C_{III0}$', r'$C_{I0}$']
xvariables_labelsmin = [r'$A_{MII0}$', r'$A_{F0}$', r'$A_{MC0}$', r'$A_{M0}$', r'$A_{M\alpha0}$', r'$T_0$', r'$I_0$', r'$C_{III0}$', r'$C_{I0}$']

# Plot max values heatmap with sorted feature importance and custom labels
plt.figure(figsize=(12, 8))
ax = sns.heatmap(max_values_df[feature_importance_max_sorted.index].reindex(columns = ["A_MC0", "A_MII0", "A_F0", "A_M0", "A_Malpha0", "beta0", "I0", "CIII0", "CI0"]), annot=True, cmap='BuGn', cbar=True)
plt.title('Max Values Heatmap (Sorted by Importance)')
plt.xlabel('Parameter Index')
plt.ylabel('Sample Index')
plt.xticks(ticks=np.arange(len(xvariables_labelsmax)) + 0.5, labels=xvariables_labelsmax, rotation=45, ha='right')

# Use row names as y-axis labels (assuming they are unique identifiers)
plt.yticks(ticks=np.arange(len(max_values_df)) + 0.5, labels=variables_labels, rotation=0)
plt.savefig('max_values_heatmap.png', dpi=300)
plt.show()

# Plot min values heatmap with sorted feature importance and custom labels
plt.figure(figsize=(12, 8))
ax = sns.heatmap(min_values_df[feature_importance_min_sorted.index].reindex(columns = ["A_MII0", "A_F0", "A_MC0", "A_M0", "A_Malpha0", "beta0", "I0", "CIII0", "CI0"]), annot=True, cmap='BuGn', cbar=True)
plt.title('Min Values Heatmap (Sorted by Importance)')
plt.xlabel('Parameter Index')
plt.ylabel('Sample Index')
plt.xticks(ticks=np.arange(len(xvariables_labelsmin)) + 0.5, labels=xvariables_labelsmin, rotation=45, ha='right')

# Use row names as y-axis labels (assuming they are unique identifiers)
plt.yticks(ticks=np.arange(len(min_values_df)) + 0.5, labels=variables_labels, rotation=0)
plt.savefig('min_values_heatmap.png', dpi=300)
plt.show()
