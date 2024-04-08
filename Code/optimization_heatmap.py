import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import MinMaxScaler
import matplotlib.ticker as ticker

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

# Sort max values to show first
max_values_df = max_values_df.sort_values(by='Value', ascending=False)

# Drop 'Label' column as we don't want to plot it
max_values_df = max_values_df.drop(columns='Label')
min_values_df = min_values_df.drop(columns='Label')

# Normalize the data using MinMaxScaler (excluding the first column)
scaler = MinMaxScaler()
max_values_df.iloc[:, 0:] = scaler.fit_transform(max_values_df.iloc[:, 0:])
min_values_df.iloc[:, 0:] = scaler.fit_transform(min_values_df.iloc[:, 0:])

variables_labels = [r'$A_{F1}$', r'$A_{F2}$', r'$A_{M1}$', r'$A_{M2}$', r'$A_{Malpha1}$', r'$A_{Malpha2}$', r'$A_{MC1}$', r'$A_{MC2}$', r'$A_{MII1}$', r'$A_{MII2}$', r'$\beta_1$', r'$\beta_2$', r'$C_{I1}$', r'$C_{I2}$', r'$C_{III1}$', r'$C_{III2}$', r'$I_{1}$', r'$I_{2}$']
xvariables_labels = [r'Value $t_{end}$', r'$A_{F}0$', r'$A_{M}0$', r'$A_{Malpha}0$', r'$A_{MC}0$', r'$A_{MII}0$', r'$\beta0$', r'$C_{I}0$', r'$C_{III}0$', r'$I0$']
# Modify y-axis labels
max_labels = ["{}".format(label) for label in variables_labels]
min_labels = ["{}".format(label) for label in variables_labels]

# Define custom colormap for max values 
cmap = LinearSegmentedColormap.from_list("custom_gnbu", sns.color_palette("BuGn", as_cmap=True)(np.linspace(0, 1, 1000)))

# Define custom colormap for min values
cmap_r = LinearSegmentedColormap.from_list("custom_gnbu_r", sns.color_palette("BuGn_r", as_cmap=True)(np.linspace(0, 1, 1000)))

# Define custom color limits for max and min values
max_vmin = 7.1e-10
max_vmax = 1.0

min_vmin = 7.7e-12
min_vmax = 1

# Plot max values heatmap with custom color limits and scientific notation color bar
plt.figure(figsize=(10, 6))
ax = sns.heatmap(max_values_df, annot=True, annot_kws={"size": 7}, cmap=cmap, cbar=True, vmin=max_vmin, vmax=max_vmax)
plt.title('Max Values Heatmap')
plt.xlabel('Parameter Index')
plt.ylabel('Sample Index')
plt.xticks(ticks=np.arange(len(xvariables_labels)) + 0.5, labels=xvariables_labels, rotation=45, ha='right')
plt.yticks(ticks=np.arange(len(max_labels)) + 0.5, labels=max_labels, rotation=0)

# Set scientific notation for color bar
formatter1 = ticker.ScalarFormatter(useMathText=True)
formatter1.set_scientific(True)
formatter1.set_powerlimits((-6, 2))  # Adjust as needed
ax.collections[0].colorbar.set_label('Normalized Value', rotation=270, labelpad=20)
ax.collections[0].colorbar.ax.yaxis.set_major_formatter(formatter1)

plt.savefig('max_values_heatmap.png', dpi=300)  # Save with DPI 300
plt.show()

# Plot min values heatmap with custom color limits and scientific notation color bar
plt.figure(figsize=(10, 6))
ax = sns.heatmap(min_values_df, annot=True, annot_kws={"size": 7}, cmap=cmap_r, cbar=True, vmin=min_vmin, vmax=min_vmax)
plt.title('Min Values Heatmap')
plt.xlabel('Parameter Index')
plt.ylabel('Sample Index')
plt.xticks(ticks=np.arange(len(xvariables_labels)) + 0.5, labels=xvariables_labels, rotation=45, ha='right')
plt.yticks(ticks=np.arange(len(min_labels)) + 0.5, labels=min_labels, rotation=0)

# Set scientific notation for color bar
formatter2 = ticker.ScalarFormatter(useMathText=True)
formatter2.set_scientific(True)
formatter2.set_powerlimits((2, -8))  # Adjust as needed
ax.collections[0].colorbar.set_label('Normalized Value', rotation=270, labelpad=20)
ax.collections[0].colorbar.ax.yaxis.set_major_formatter(formatter2)

plt.savefig('min_values_heatmap.png', dpi=300)  # Save with DPI 300
plt.show()
