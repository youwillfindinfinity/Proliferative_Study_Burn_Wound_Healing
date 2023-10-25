from correlationmatrix import run_simulation
# from param_ranges import p_range
import json
import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
from params import initial_parameters
import sys
sys.setrecursionlimit(10000)

ranges_for_production = [0.001, 5]
ranges_for_upsi = [0.001, 0.1]
ranges_for_omega = [0, 2 * np.pi]
ranges_for_gamma = [1.0 * 10**-7, 1.0 * 10**-5]
ranges_for_zeta = [1.0 * 10**-7, 1.0 * 10**-1]
ranges_for_mu = [0.0001, 1]
ranges_for_lambdas = ranges_for_mu
ranges_for_rhos = ranges_for_mu

# Define parameter ranges for sensitivity analysis
p_ranges = {
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
  'upsilon1': (-1, 0),
  'upsilon2': (-1, 1),
  'upsilon3': ranges_for_upsi,
  'upsilon4': ranges_for_upsi,
  'lambda1': ranges_for_lambdas,
  'lambda2': ranges_for_lambdas,
  'lambda3': ranges_for_lambdas,
  'lambda4': ranges_for_lambdas,
  'rho1': ranges_for_rhos,
  'rho2': ranges_for_rhos,
  'rho3': ranges_for_rhos,
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
  'A_MC0': (0, 3000),
  'A_F0': (500, 1500),
  'A_M0': (400, 1200),
  'A_Malpha0':(0, 0.000001),
   'CI0':(0, 0.000001),
   'CIII0':(0, 0.000001),
   'f_dillution':(1/16, 1/15),
   "gamma":(10**-7, 10**-5),
   "zeta":(10**-7, 10**-5),
   'omega1':(0, 2 * np.pi),
   'omega2':(0, 2 * np.pi),
   'omega3':(0, 2 * np.pi),
}



# # Function to sample parameters within specified ranges
# def sample_parameters_within_ranges():
#     sampled_parameters = {}
#     for param, (low, high) in p_ranges.items():
#         sampled_value = 0
#         # Sample within the specified range
#         if param in ['A_F0', 'A_F1', 'A_F2']:
#             sampled_value = np.random.uniform(max(low, 500), min(high, 1500))
#         elif param in ['A_M0', 'A_M1', 'A_M2']:
#             sampled_value = np.random.uniform(max(low, 400), min(high, 1200))
#         elif param in ['A_Malpha0', 'CI0', 'CIII0']:
#             sampled_value = np.random.uniform(max(low, 0), min(high, 0.000001))
#         else:
#             sampled_value = np.random.uniform(low, high)
#         sampled_parameters[param] = sampled_value
#     return sampled_parameters
# # Function to run the simulation and validate the results
# def calc_sim(parameters):
#  # Call the simulation function to get results
#     results = run_simulation(parameters)

#     # Validate simulation results
#     valid_simulation = all(0 <= result <= 4000 for sublist in results for result in sublist)

#     return results, valid_simulation




# initial_parameters = initial_parameters()

# # Create a directory to store Monte Carlo results
# output_directory = "monte_carlo_results"
# os.makedirs(output_directory, exist_ok=True)


# # Number of Monte Carlo simulations
# num_simulations = 10000

# # List to store the best simulations
# best_simulations = []
# # Perform Monte Carlo simulations
# for simulation_index in range(num_simulations):
#     valid_simulation = False
#     sampled_parameters = None
#     simulation_results = None
    
#     # Evaluate simulations until a valid one is found
#     while not (valid_simulation) and (simulation_index != num_simulations):
#         # Sample parameters within the specified ranges
#         sampled_parameters = sample_parameters_within_ranges()

#         # Run the simulation with the sampled parameters
#         simulation_results, valid_simulation = calc_sim(sampled_parameters)
    
#     # Check if the current simulation has higher values at the end for each list
#     if not best_simulations or all(result[-1] >= best_result[-1] for result, best_result in zip(simulation_results, best_simulations)):
#         best_simulations = simulation_results
#     if simulation_index == num_simulations:
#     	break
#     # Break the loop if 10 best simulations are found
#     if len(best_simulations) >= 10:
#         break

# # Export the best 10 simulation parameters to a JSON file
# best_simulation_parameters = []
# for simulation_results in best_simulations:
#     # Extract parameters from the simulation_results if needed
#     parameters = extract_parameters(simulation_results)  # Modify this according to your simulation structure
#     best_simulation_parameters.append(parameters)

# # Write the best 10 simulation parameters to a JSON file
# with open("best_simulation_parameters2.json", "w") as json_file:
#     json.dump(best_simulation_parameters, json_file)


# print(best_simulation_parameters)
# # Extracted parameter values from the simulation results


# Plotting the results
# plt.plot(time, a)
# plt.plot(time, b)
# plt.plot(time, c)
plt.plot(b)
# Adjust layout and show the plots
plt.tight_layout()
plt.show()

# # Run the simulation with the extracted parameters
# # Time parameters
# weeks = 30
# n_days_in_week = 7
# t_max =  weeks * n_days_in_week # Maximum simulation time(weeks)
# dt = weeks/t_max # Time step

# # Forward Euler method
# timesteps = int(t_max / dt)
# time = np.linspace(0, t_max, timesteps)


# a, b, c, d, e, f, g, h = calc_sim(parameters)
# plt.plot(time, a)
# # Adjust layout and show the plots
# plt.tight_layout()
# plt.show()