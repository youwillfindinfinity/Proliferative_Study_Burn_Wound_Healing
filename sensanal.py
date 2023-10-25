from params import initial_parameters
from param_ranges import parameter_ranges
from main import A_MII1_func,A_MII2_func,A_Malpha_func, scenario1_equations, scenario2_equations
from scipy.interpolate import griddata
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

# # Counter for JSON file names
# json_file_counter = 1

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
#                 sensitivity_metric = run_simulation(initial_parameters) 
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

#                 # Save the sensitivity results to a new JSON file
#                 with open(f'sensitivity_results/sensitivity_results{json_file_counter}.json', 'w') as json_file:
#                     json.dump(sensitivity_results, json_file)

#                 # Increment the JSON file counter
#                 json_file_counter += 1

#         # Reset the parameters to their initial values for the next iteration
#         initial_parameters[param_x] = initial_value_x
#         initial_parameters[param_y] = initial_value_y


# print("Sensitivity results have been saved to 'sensitivity_results.json'.")
import json
import matplotlib.pyplot as plt

# Load sensitivity results from the JSON file
with open('sensitivity_results/sensitivity_results1675.json', 'r') as json_file:
    sensitivity_results = json.load(json_file)

# Extract data from sensitivity results
x_values = sensitivity_results['x_values']
y_values = sensitivity_results['y_values']
# A_F1 = sensitivity_results['sensitivity_metrics']['A_F1'][:len(x_values)]
# List of keys from the dictionary
keys = list(sensitivity_results['sensitivity_metrics'].keys())

# Create a figure with 8 subplots arranged in 2 rows and 4 columns
plt.figure(figsize=(16, 8))

for idx, key in enumerate(keys):
    # Trim the data to match the length of x_values
    data = sensitivity_results['sensitivity_metrics'][key][:len(x_values)]

    # Create the meshgrid
    xi, yi = np.meshgrid(x_values, y_values)

    # Convert data to a NumPy array and flatten it
    data_array = np.array(data)
    flat_data = data_array.ravel()

    # Construct flattened x_values and y_values
    flat_x_values = np.tile(np.linspace(min(x_values), max(x_values), data_array.shape[1]), data_array.shape[0])
    flat_y_values = np.repeat(np.linspace(min(y_values), max(y_values), data_array.shape[0]), data_array.shape[1])

    # Reshape the data to match the grid shape
    data_reshaped = griddata((flat_x_values, flat_y_values), flat_data, (xi, yi), method='cubic')

    # Fill NaN and Inf values with 0
    data_filled = np.nan_to_num(data_reshaped, nan=0.0001, posinf=0.0001, neginf=0.0001)
    data_filled[data_filled == 0] = 0.001


    # Plot the contour plot in the corresponding subplot
    plt.subplot(2, 4, idx + 1)
    plt.contourf(xi, yi, data_filled, levels=20, cmap='viridis')
    plt.colorbar(label=key)
    plt.xlabel('Parameters within range')
    plt.ylabel('Parameters within range')
    plt.title(f'Contour Plot of {key}')

# Adjust layout
plt.tight_layout()

# Save the figure as an image file (e.g., PNG, PDF, etc.)
plt.savefig('contour_plots.png')  # You can change the file format and file name as needed

# Show the plot (optional, depending on your use case)
plt.show()