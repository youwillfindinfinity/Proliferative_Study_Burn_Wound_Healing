import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

from main import *
from params import defined_params
from param_ranges import *
from scipy.optimize import approx_fprime

# Define the parameters to be excluded
excluded_parameters = ['k1', 'k2', 'k4', 'k7', 'k8', 'k11', 'lambda1', \
'rho1', 'mu1', 'mu2', 'mu5', 'mu7', 'mu8', 'gamma1', 'gamma2', 'gamma3',\
 'gamma4', 'gamma5', 'gamma6', 'gamma7', 'gamma8', 'zeta1', 'zeta2', 'zeta3', \
 'zeta4', 'zeta5', 'f_dillution']

ranges_for_production = (0.001, 5)
ranges_for_upsi = (0.001, 0.1)
ranges_for_omega = (0, 2 * np.pi)
ranges_for_gamma = (1.0 * 10**-7, 1.0 * 10**-5)
ranges_for_zeta = (1.0 * 10**-7, 1.0 * 10**-1)
ranges_for_mu = (0.0001, 1)

def parameters_valid(parameters):
    # Define the valid ranges for each parameter
    valid_ranges = {
      'k1': ranges_for_production,
      'k2': ranges_for_production,
      'k3': ranges_for_production,
      'k4': ranges_for_production,
      'k5': ranges_for_production,
      'k6': ranges_for_production,
      'k7': ranges_for_production,
      'k8': ranges_for_production,
      'k9': ranges_for_production,
      'k10': ranges_for_production,
      'k11': ranges_for_production,
      'upsilon1': ranges_for_upsi,
      'upsilon2': ranges_for_upsi,
      'upsilon3': ranges_for_upsi,
      'upsilon4': ranges_for_upsi,
      'omega1': ranges_for_omega,
      'omega2': ranges_for_omega,
      'omega3': ranges_for_omega,
      'gamma1': ranges_for_gamma,
      'gamma2': ranges_for_gamma,
      'gamma3': ranges_for_gamma,
      'gamma4': ranges_for_gamma,
      'gamma5': ranges_for_gamma,
      'gamma6': ranges_for_gamma,
      'gamma7': ranges_for_gamma,
      'gamma8': ranges_for_gamma,
      'zeta1': ranges_for_zeta,
      'zeta2': ranges_for_zeta,
      'zeta3': ranges_for_zeta,
      'zeta4': ranges_for_zeta,
      'zeta5': ranges_for_zeta,
      'mu1': ranges_for_mu,
      'mu2': ranges_for_mu,
      'mu3': ranges_for_mu,
      'mu4': ranges_for_mu,
      'mu5': ranges_for_mu,
      'mu6': ranges_for_mu,
      'mu7': ranges_for_mu,
      'mu8': ranges_for_mu,
      'A_MII0': (1000, 3000),
      'I0': (10**-9, 10**-6),
      'beta0': (10**-9, 10**-6),
      'A_MC0': (0, 500),
      'A_F0': (0, 500),
      'A_M0': (0, 200)
    }

    # Check if each parameter is within its valid range
    for param, (min_val, max_val) in valid_ranges.items():
        if param in parameters and not (min_val <= parameters[param] <= max_val):
            return False  # Invalid parameter value found

    return True  # All parameters are valid


# Function to calculate model output for given parameter values
def model_function(parameters, parameters_copy, scenario_nr):

    # Unpack non-defined keys from parameters dictionary into individual variables
    for key in parameters:
        if key not in locals():
            locals()[key] = parameters[key]

     # Unpack non-defined keys from parameters dictionary into individual variables
    for key in parameters_copy:
        locals()[key] = parameters_copy[key]



    # Time parameters
    weeks = 30
    n_days_in_week = 7
    t_max =  weeks * n_days_in_week # Maximum simulation time(weeks)
    dt = weeks/t_max # Time step

    # Forward Euler method
    timesteps = int(t_max / dt)
    time = np.linspace(0, t_max, timesteps)

    # Perform simulation for both scenarios using forward Euler method
    if scenario_nr == "1":
    # Initialize arrays for results
        A_MII1 = [A_MII0]
        I1 = [I0]
        beta1 = [beta0]
        A_MC1 = [A_MC0]
        A_F1 = [A_F0]
        A_M1 = [A_M0]
        A_Malpha1 = [A_Malpha0]
        CIII1 = [CIII0]
        CI1 = [CI0]

        for i in range(1, timesteps + 1):
            t = i * dt
            
            # Scenario 1
            A_MII_next, I_next, beta_next, A_MC_next, A_F_next, A_M_next, A_Malpha_next, CIII_next, CI_next = \
                scenario1_equations(A_MII1[-1], I1[-1], beta1[-1], A_MC1[-1], A_F1[-1], A_M1[-1], A_Malpha1[-1], CIII1[-1], CI1[-1], t)
            A_MII1.append(A_MII_next)
            I1.append(I_next)
            beta1.append(beta_next)
            A_MC1.append(A_MC_next)
            A_F1.append(A_F_next)
            A_M1.append(A_M_next)
            A_Malpha1.append(A_Malpha_next)
            CIII1.append(CIII_next)
            CI1.append(CI_next)
        output_variables = I1[1:], beta1[1:], A_MII1[1:], A_MC1[1:], A_F1[1:], A_M1[1:], CI1[1:], CIII1[1:], A_Malpha1[1:],
        

    if scenario_nr == "2":
        A_MII2 = [A_MII0]
        I2 = [I0]
        beta2 = [beta0]
        A_MC2 = [A_MC0]
        A_F2 = [A_F0]
        A_M2 = [A_M0]
        A_Malpha2 = [A_Malpha0]
        CIII2 = [CIII0]
        CI2 = [CI0]

        
        for i in range(1, timesteps + 1):
            t = i * dt
             # Scenario 2
            A_MII_next, I_next, beta_next, A_MC_next, A_F_next, A_M_next, A_Malpha_next, CIII_next, CI_next = \
                scenario2_equations(A_MII2[-1], I2[-1], beta2[-1], A_MC2[-1], A_F2[-1], A_M2[-1], A_Malpha2[-1], CIII2[-1], CI2[-1], t)
            A_MII2.append(A_MII_next)
            I2.append(I_next)
            beta2.append(beta_next)
            A_MC2.append(A_MC_next)
            A_F2.append(A_F_next)
            A_M2.append(A_M_next)
            A_Malpha2.append(A_Malpha_next)
            CIII2.append(CIII_next)
            CI2.append(CI_next)
        
        output_variables = I2[1:], beta2[1:], A_MII2[1:], A_MC2[1:], A_F2[1:], A_M2[1:], CI2[1:], CIII2[1:], A_Malpha2[1:]
    return output_variables



def simulation(parameter_ranges, scenario_nr):
    # Initialize empty lists to store results
    results = {param: [] for param in parameter_ranges}
    # Initialize an empty list to store parameter and result pairs
    parameter_result_pairs = []

    # Perform sensitivity analysis
    for param, values in parameter_ranges.items():

        if param in excluded_parameters:
            continue

        for value in values:
            # Create a copy of your original parameters and initial conditions
            parameters_copy = {
                'k1': 2.34 * 10**-6,
                'k2': 234 * 10**-5 * day_conversion,
                'k3': 0.15,
                'k4': 280 * 10**-5 * day_conversion,
                'k5': 0.2,
                'k6': 0.3,
                'k7': 50 * 10**-5,
                'k8': 30 * 10**-5,
                'k9': 0.23,
                'k10': 0.1,
                'k11': 2 * 10**(-7) * day_conversion,
                'upsilon1': 0.02,
                'upsilon2': 0.03,
                'upsilon3': 0.01,
                'upsilon4': 0.02,
                'omega1': 0.5,
                'omega2': 0.7,
                'omega3': 0.6,
                'gamma1': 10**(-5),
                'gamma2': 10**(-5),
                'gamma3': 10**(-5),
                'gamma4': 10**(-5),
                'gamma5': 10**(-5),
                'gamma6': 10**(-5),
                'gamma7': 10**(-5),
                'gamma8': 10**(-5),
                'zeta1': 10**(-5),
                'zeta2': 10**(-5),
                'zeta3': 10**(-5),
                'zeta4': 10**(-5),
                'zeta5': 10**(-5),
                'mu1': 0.07,
                'mu2': 7,
                'mu3': 0.03,
                'mu4': 0.01,
                'mu5': 0.1,
                'mu6': 0.03,
                'mu7': 9.7 * 10**(-5) * day_conversion,
                'mu8': 9.7 * 10**(-5) * day_conversion,
                'A_MII0': 2000,
                'I0': 10**(-9),
                'beta0': 10**(-7),
                'A_MC0': 100,
                'A_F0': 500,
                'A_M0': 20,
                'A_Malpha0': 0,
                'CIII0': 0,
                'CI0': 0
            }
            # Set the current parameter value
            parameters_copy[param] = value
            # Run the simulation with modified parameter

            # Time parameters
            weeks = 30
            n_days_in_week = 7
            t_max =  weeks * n_days_in_week # Maximum simulation time(weeks)
            dt = weeks/t_max # Time step

            # Forward Euler method
            timesteps = int(t_max / dt)
            time = np.linspace(0, t_max, timesteps)


            # Perform simulation for both scenarios using forward Euler method
            if scenario_nr == "1":
            # Initialize arrays for results
                A_MII1 = [A_MII0]
                I1 = [I0]
                beta1 = [beta0]
                A_MC1 = [A_MC0]
                A_F1 = [A_F0]
                A_M1 = [A_M0]
                A_Malpha1 = [A_Malpha0]
                CIII1 = [CIII0]
                CI1 = [CI0]

                for i in range(1, timesteps + 1):
                    t = i * dt
                    
                    # Scenario 1
                    A_MII_next, I_next, beta_next, A_MC_next, A_F_next, A_M_next, A_Malpha_next, CIII_next, CI_next = \
                        scenario1_equations(A_MII1[-1], I1[-1], beta1[-1], A_MC1[-1], A_F1[-1], A_M1[-1], A_Malpha1[-1], CIII1[-1], CI1[-1], t)
                    A_MII1.append(A_MII_next)
                    I1.append(I_next)
                    beta1.append(beta_next)
                    A_MC1.append(A_MC_next)
                    A_F1.append(A_F_next)
                    A_M1.append(A_M_next)
                    A_Malpha1.append(A_Malpha_next)
                    CIII1.append(CIII_next)
                    CI1.append(CI_next)
                output_variables = I1[1:], beta1[1:], A_MII1[1:], A_MC1[1:], A_F1[1:], A_M1[1:], CI1[1:], CIII1[1:], A_Malpha1[1:],
                

            if scenario_nr == "2":
                A_MII2 = [A_MII0]
                I2 = [I0]
                beta2 = [beta0]
                A_MC2 = [A_MC0]
                A_F2 = [A_F0]
                A_M2 = [A_M0]
                A_Malpha2 = [A_Malpha0]
                CIII2 = [CIII0]
                CI2 = [CI0]

                
                for i in range(1, timesteps + 1):
                    t = i * dt
                     # Scenario 2
                    A_MII_next, I_next, beta_next, A_MC_next, A_F_next, A_M_next, A_Malpha_next, CIII_next, CI_next = \
                        scenario2_equations(A_MII2[-1], I2[-1], beta2[-1], A_MC2[-1], A_F2[-1], A_M2[-1], A_Malpha2[-1], CIII2[-1], CI2[-1], t)
                    A_MII2.append(A_MII_next)
                    I2.append(I_next)
                    beta2.append(beta_next)
                    A_MC2.append(A_MC_next)
                    A_F2.append(A_F_next)
                    A_M2.append(A_M_next)
                    A_Malpha2.append(A_Malpha_next)
                    CIII2.append(CIII_next)
                    CI2.append(CI_next)
                
                output_variables = I2[1:], beta2[1:], A_MII2[1:], A_MC2[1:], A_F2[1:], A_M2[1:], CI2[1:], CIII2[1:], A_Malpha2[1:]
            # Store parameters and results as a dictionary
            iteration_results = {
                'parameters': parameters_copy,
                'results': output_variables
            }
            
            # Append the parameter and result pair to the list
            parameter_result_pairs.append(iteration_results)

    # Check if the file exists and append data to it if it does, else create a new file
    file_path = 'parameter_results_{}.json'.format(scenario_nr)
    mode = 'a' if os.path.exists(file_path) else 'w'

    with open(file_path, mode) as file:
        for pair in parameter_result_pairs:
            json.dump(pair, file)
            file.write('\n')

    # Check if the file 'parameter_results.json' exists
    if os.path.exists('parameter_results_{}.json'.format(scenario_nr)):
        # Read data from the file and update parameter_result_pairs
        with open('parameter_results_{}.json'.format(scenario_nr), 'r') as file:
            lines = file.readlines()
            for line in lines:
                pair = json.loads(line)
                parameter_result_pairs.append(pair)

    # Extract simulation results as arrays for reshaping
    sim_results_array = np.array([np.array(pair['results']) for pair in parameter_result_pairs])

    # Flatten the 3D array sim_results_array into a 2D array for correlation calculation
    sim_results_flat = sim_results_array.reshape(sim_results_array.shape[0], -1)

    # Calculate the correlation matrix with handling for zero standard deviation
    with np.errstate(divide='ignore', invalid='ignore'):
        correlation_matrix = np.corrcoef(sim_results_flat, rowvar=False)

    # Replace NaN values with 0 in the correlation matrix
    correlation_matrix = np.nan_to_num(correlation_matrix)

    # Save the correlation matrix to a JSON file
    with open('correlation_matrix_{}.json'.format(scenario_nr), 'w') as file:
        json.dump(correlation_matrix.tolist(), file)
    return



def local_sensitivity_analysis(parameter_ranges, excluded_parameters, scenario_nr):
    # Initialize empty dictionary to store sensitivity coefficients
    sensitivity_coefficients = {param: [] for param in parameter_ranges}

    # Define a small perturbation for finite difference approximation
    epsilon = 1e-2

    # Perform sensitivity analysis for each parameter
    for param in parameter_ranges:
        if param in excluded_parameters:
            continue
        for value in parameter_ranges[param]:
            # Create a copy of your original parameters and initial conditions
            parameters_copy = {
                'k1': 2.34 * 10**-6,
                'k2': 234 * 10**-5 * day_conversion,
                'k3': 0.15,
                'k4': 280 * 10**-5 * day_conversion,
                'k5': 0.2,
                'k6': 0.3,
                'k7': 50 * 10**-5,
                'k8': 30 * 10**-5,
                'k9': 0.23,
                'k10': 0.1,
                'k11': 2 * 10**(-7) * day_conversion,
                'upsilon1': 0.02,
                'upsilon2': 0.03,
                'upsilon3': 0.01,
                'upsilon4': 0.02,
                'omega1': 0.5,
                'omega2': 0.7,
                'omega3': 0.6,
                'gamma1': 10**(-5),
                'gamma2': 10**(-5),
                'gamma3': 10**(-5),
                'gamma4': 10**(-5),
                'gamma5': 10**(-5),
                'gamma6': 10**(-5),
                'gamma7': 10**(-5),
                'gamma8': 10**(-5),
                'zeta1': 10**(-5),
                'zeta2': 10**(-5),
                'zeta3': 10**(-5),
                'zeta4': 10**(-5),
                'zeta5': 10**(-5),
                'mu1': 0.07,
                'mu2': 7,
                'mu3': 0.03,
                'mu4': 0.01,
                'mu5': 0.1,
                'mu6': 0.03,
                'mu7': 9.7 * 10**(-5) * day_conversion,
                'mu8': 9.7 * 10**(-5) * day_conversion,
                'A_MII0': 2000,
                'I0': 10**(-9),
                'beta0': 10**(-7),
                'A_MC0': 100,
                'A_F0': 500,
                'A_M0': 20,
                'A_Malpha0': 0,
                'CIII0': 0,
                'CI0': 0
            }
            parameters_copy[param] = value

            # Calculate model output for the base parameters
            base_output = np.array(model_function(parameters_copy, parameters_copy, scenario_nr))

            # Perturb the parameter and calculate model output for the perturbed parameters
            parameters_copy[param] += epsilon
            perturbed_output = np.array(model_function(parameters_copy ,parameters_copy, scenario_nr))

            # Calculate sensitivity coefficient using finite difference approximation
            sensitivity_coefficient = (perturbed_output - base_output) / epsilon

            # Store sensitivity coefficient for the current parameter and value combination
            sensitivity_coefficients[param].append(sensitivity_coefficient.tolist())

    # Save sensitivity coefficients to a JSON file
    with open('sensitivity_coefficients_{}.json'.format(scenario_nr), 'w') as file:
        json.dump(sensitivity_coefficients, file)

    return sensitivity_coefficients



def plot(scenario_nr):
    # Read the correlation matrix from the JSON file
    with open('correlation_matrix_{}.json'.format(scenario_nr), 'r') as file:
        correlation_matrix_json = json.load(file)

    # Convert the correlation matrix from a nested list to a NumPy array
    correlation_matrix = np.array(correlation_matrix_json)

    # Get the parameter names excluding excluded_parameters
    parameter_names = [param for param in parameter_ranges.keys() if param not in excluded_parameters]

    # Create a clustermap with dendrogram using only the included parameters
    plt.figure(figsize=(12, 10))
    sns.set(font_scale=1.0)  # Adjust font size if needed
    sns.clustermap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f", xticklabels=parameter_names, yticklabels=parameter_names)

    # Set plot title and labels
    plt.title('Correlation Matrix')
    plt.xlabel('Parameters')
    plt.ylabel('Parameters')

    # Save the correlation matrix clustermap as an image file
    plt.savefig('correlation_matrix_{}.png'.format(scenario_nr))

    # Show the correlation matrix clustermap
    # plt.show()

def plot_sensitivity_coefficients(scenario_nr):
    # Read sensitivity coefficients from the JSON file
    with open('sensitivity_coefficients_{}.json'.format(scenario_nr), 'r') as file:
        sensitivity_coefficients = json.load(file)

    # Get the parameter names excluding excluded_parameters
    parameter_names = [param for param in sensitivity_coefficients.keys()]

    # Convert sensitivity coefficients to a list
    sensitivity_values = [sensitivity_coefficients[param] for param in parameter_names]

    # Create a bar plot for sensitivity coefficients
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sensitivity_values, y=parameter_names, palette='coolwarm')
    plt.xlabel('Sensitivity Coefficients')
    plt.ylabel('Parameters')
    plt.title('Sensitivity Coefficients')
    plt.tight_layout()

    # Save the sensitivity coefficients plot as an image file
    plt.savefig('sensitivity_coefficients_{}.png'.format(scenario_nr))

    # Show the sensitivity coefficients plot
    # plt.show()
# simulation(parameter_ranges, scenario_nr = "1")
# simulation(parameter_ranges, scenario_nr = "2")
local_sensitivity_analysis(parameter_ranges, excluded_parameters, "1")
# plot(scenario_nr = "1")
# plot_sensitivity_coefficients("1")
