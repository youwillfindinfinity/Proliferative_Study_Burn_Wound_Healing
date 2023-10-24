from params import initial_parameters
from param_ranges import parameter_ranges
from main import A_MII1_func,A_MII2_func,A_Malpha_func, scenario1_equations, scenario2_equations
import numpy as np
import matplotlib.pyplot as plt
import json
# # Define your simulation function here (using the initial_parameters)
# def run_simulation(parameters):
#     # Time parameters
#     weeks = 30
#     n_days_in_week = 7
#     t_max =  weeks * n_days_in_week # Maximum simulation time(weeks)
#     dt = weeks/t_max # Time step

#     # Forward Euler method
#     timesteps = int(t_max / dt)
#     time = np.linspace(0, t_max, timesteps)


#     # Initialize arrays for results
#     A_MII1 = [parameters['A_MII0']]
#     I1 = [parameters['I0']]
#     beta1 = [parameters['beta0']]
#     A_MC1 = [parameters['A_MC0']]
#     A_F1 = [parameters['A_F0']]
#     A_M1 = [parameters['A_M0']]
#     A_Malpha1 = [parameters['A_Malpha0']]
#     CIII1 = [parameters['CIII0']]
#     CI1 = [parameters['CI0']]

#     # Initialize arrays for results
#     A_MII2 = [parameters['A_MII0']]
#     I2 = [parameters['I0']]
#     beta2 = [parameters['beta0']]
#     A_MC2 = [parameters['A_MC0']]
#     A_F2 = [parameters['A_F0']]
#     A_M2 = [parameters['A_M0']]
#     A_Malpha2 = [parameters['A_Malpha0']]
#     CIII2 = [parameters['CIII0']]
#     CI2 = [parameters['CI0']]

#     # Perform simulation for both scenarios using forward Euler method
#     for i in range(1, timesteps + 1):
#         t = i * dt
        
#         # Scenario 1
#         A_MII_next, I_next, beta_next, A_MC_next, A_F_next, A_M_next, A_Malpha_next, CIII_next, CI_next = \
#             scenario1_equations(A_MII1[-1], I1[-1], beta1[-1], A_MC1[-1], A_F1[-1], A_M1[-1], A_Malpha1[-1], CIII1[-1], CI1[-1], parameters, dt, t)
#         A_MII1.append(A_MII_next)
#         I1.append(I_next)
#         beta1.append(beta_next)
#         A_MC1.append(A_MC_next)
#         A_F1.append(A_F_next)
#         A_M1.append(A_M_next)
#         A_Malpha1.append(A_Malpha_next)
#         CIII1.append(CIII_next)
#         CI1.append(CI_next)
        
#         # Scenario 2
#         A_MII_next, I_next, beta_next, A_MC_next, A_F_next, A_M_next, A_Malpha_next, CIII_next, CI_next = \
#             scenario2_equations(A_MII2[-1], I2[-1], beta2[-1], A_MC2[-1], A_F2[-1], A_M2[-1], A_Malpha2[-1], CIII2[-1], CI2[-1], parameters, dt, t)
#         A_MII2.append(A_MII_next)
#         I2.append(I_next)
#         beta2.append(beta_next)
#         A_MC2.append(A_MC_next)
#         A_F2.append(A_F_next)
#         A_M2.append(A_M_next)
#         A_Malpha2.append(A_Malpha_next)
#         CIII2.append(CIII_next)
#         CI2.append(CI_next)
#     return A_F1[1:], A_M1[1:], CIII1[1:], CI1[1:], A_F2[1:], A_M2[1:], CIII2[1:], CI2[1:]

# initial_parameters = initial_parameters()


# # Initialize empty dictionaries to store sensitivity metrics and parameter values
# sensitivity_results = {'x_values': [], 'y_values': [], 'sensitivity_metrics': {'A_F1':[], 
# 'A_M1':[], 'CIII1':[], 'CI1':[], 'A_F2':[], 'A_M2':[], 'CIII2':[], 'CI2':[]}}

# # Perform sensitivity analysis for each parameter
# for param_x, param_values_x in parameter_ranges.items():
#     for param_y, param_values_y in parameter_ranges.items():
#         initial_value_x = initial_parameters[param_x]
#         initial_value_y = initial_parameters[param_y]
#         for value_x in param_values_x:
#             for value_y in param_values_y:
#                 # Modify the parameters in initial_parameters with the new values
#                 initial_parameters[param_x] = value_x
#                 initial_parameters[param_y] = value_y
#                 # Run simulation function with updated parameters and get the sensitivity metric
#                 sensitivity_metric = run_simulation(initial_parameters)  # Modify this line based on your simulation function
#                 # Store the results in the dictionary
#                 sensitivity_results['x_values'].append(value_x)
#                 sensitivity_results['y_values'].append(value_y)
#                 sensitivity_results['sensitivity_metrics']['A_F1'].append(sensitivity_metric[0])
#                 sensitivity_results['sensitivity_metrics']['A_M1'].append(sensitivity_metric[1])
#                 sensitivity_results['sensitivity_metrics']['CIII1'].append(sensitivity_metric[2])
#                 sensitivity_results['sensitivity_metrics']['CI1'].append(sensitivity_metric[3])
#                 sensitivity_results['sensitivity_metrics']['A_F2'].append(sensitivity_metric[4])
#                 sensitivity_results['sensitivity_metrics']['A_M2'].append(sensitivity_metric[5])
#                 sensitivity_results['sensitivity_metrics']['CIII2'].append(sensitivity_metric[6])
#                 sensitivity_results['sensitivity_metrics']['CI2'].append(sensitivity_metric[7])

#         # Reset the parameters to their initial values for the next iteration
#         initial_parameters[param_x] = initial_value_x
#         initial_parameters[param_y] = initial_value_y

# # Save the sensitivity results to a JSON file
# with open('sensitivity_results1.json', 'w') as json_file:
#     json.dump(sensitivity_results, json_file)

# print("Sensitivity results have been saved to 'sensitivity_results.json'.")

# Load sensitivity analysis results from the JSON file
with open('sensitivity_results.json', 'r') as json_file:
    sensitivity_results = json.load(json_file)

# List of sensitivity metrics to plot
sensitivity_metrics_list = ['A_F1', 'A_M1', 'CIII1', 'CI1', 'A_F2', 'A_M2', 'CIII2', 'CI2']

# Number of rows and columns for subplots
num_rows = 2
num_cols = 4

# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 8))

# Flatten and plot each sensitivity metric
for i, metric in enumerate(sensitivity_metrics_list):
    row = i // num_cols
    col = i % num_cols
    
    # Extract and flatten the sensitivity metric data
    sensitivity_metric_flat = np.ravel(sensitivity_results['sensitivity_metrics'][metric])
    
    # Plot the contour plot for the current metric
    contour = axes[row, col].tricontourf(sensitivity_results['x_values'], sensitivity_results['y_values'], sensitivity_metric_flat, levels=20, cmap='viridis')
    axes[row, col].set_xlabel('Parameter X Values')
    axes[row, col].set_ylabel('Parameter Y Values')
    axes[row, col].set_title(metric)
    fig.colorbar(contour, ax=axes[row, col], label='Sensitivity Metric')

    # Save the individual figure for the current metric
    individual_fig_name = f'{metric}_sensitivity_plot.png'
    plt.savefig(individual_fig_name)
    print(f"Saved: {individual_fig_name}")

# Adjust layout and display the plots
plt.tight_layout()
plt.show()

# Save the joint figure
joint_fig_name = 'joint_sensitivity_plots.png'
plt.savefig(joint_fig_name)
print(f"Saved: {joint_fig_name}")

# # Time parameters
# weeks = 30
# n_days_in_week = 7
# t_max =  weeks * n_days_in_week # Maximum simulation time(weeks)
# dt = weeks/t_max # Time step

# # Forward Euler method
# parameters = initial_parameters()
# timesteps = int(t_max / dt)
# time = np.linspace(0, t_max, timesteps)
# sim = run_simulation(parameters)
# print(sim[0])
# plt.plot(time, a)
# plt.plot(time, b)
# plt.show()
