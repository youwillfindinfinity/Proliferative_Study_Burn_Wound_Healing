import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the functions for fitting
def fit_function(x, a, b, c):
    return a * x**b + c

# Define the weeks and corresponding states for cells
weeks = np.arange(0, 6)  # 0 to 5 weeks
cell_states = {
    'Fibroblasts': ['↑', '→', '→', '↓', '↓', '↓'],
    'Myofibroblasts': ['→', '→', '↑', '→', '↓', '↓'],
    'Alpha-Smooth Muscle Actin (αSMA)': ['↑', '→', '→', '→', '↓', '↓'],
    'Platelets': ['↑', '→', '→', '→', '↓', '↓']
}

# Function to map symbols to numerical values for fitting
def map_symbols_to_numbers(states):
    mapping = {'↑': 1, '→': 0, '↓': -1}
    return [mapping[state] for state in states]

# Convert states to numerical values for fitting
for cell_type, states in cell_states.items():
    cell_states[cell_type] = map_symbols_to_numbers(states)

# Data for fitting
x_data = weeks

# Create subplots for cell dynamics
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Cell Dynamics', fontsize=16)

# Fit functions and plot cell dynamics
for i, (cell_type, states) in enumerate(cell_states.items()):
    row = i // 2
    col = i % 2
    popt, _ = curve_fit(fit_function, x_data, states)
    axes[row, col].plot(x_data, states, 'o', label='Data')
    axes[row, col].plot(x_data, fit_function(x_data, *popt), '--', label='Fit')
    axes[row, col].set_title(f'{cell_type} Dynamics')
    axes[row, col].set_xlabel('Weeks')
    axes[row, col].set_ylabel('State')
    axes[row, col].set_xticks(weeks)
    axes[row, col].set_yticks([-1, 0, 1])
    axes[row, col].set_yticklabels(['↓', '→', '↑'])
    axes[row, col].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# Define the weeks and corresponding states for cytokines
cytokine_states = {
    'IL-8': ['↑', '→', '→', '↑', '↓', '↓'],
    'PDGF': ['↑', '→', '→', '↑', '↓', '↓'],
    'TGF-β': ['↑', '→', '→', '→', '↓', '↓'],
    'EGF': ['↑', '→', '→', '→', '↓', '↓']
}

# Convert states to numerical values for fitting
for cytokine, states in cytokine_states.items():
    cytokine_states[cytokine] = map_symbols_to_numbers(states)

# Data for fitting
x_data = weeks

# Create subplots for cytokine dynamics
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Cytokine Dynamics', fontsize=16)

# Fit functions and plot cytokine dynamics
for i, (cytokine, states) in enumerate(cytokine_states.items()):
    row = i // 2
    col = i % 2
    popt, _ = curve_fit(fit_function, x_data, states)
    axes[row, col].plot(x_data, states, 'o', label='Data')
    axes[row, col].plot(x_data, fit_function(x_data, *popt), '--', label='Fit')
    axes[row, col].set_title(f'{cytokine} Dynamics')
    axes[row, col].set_xlabel('Weeks')
    axes[row, col].set_ylabel('State')
    axes[row, col].set_xticks(weeks)
    axes[row, col].set_yticks([-1, 0, 1])
    axes[row, col].set_yticklabels(['↓', '→', '↑'])
    axes[row, col].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
