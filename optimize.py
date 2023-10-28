import numpy as np
import json
from scipy.optimize import minimize, curve_fit
from params import initial_parameters, defined_params
from param_ranges import p_ranges
from main import A_MII1_func,A_MII2_func,A_Malpha_func, scenario1_equations, scenario2_equations
import matplotlib.pyplot as plt



def run_simulation(params):
    # print(parameters)
    # k3, k5, k6, k9, k10, lambda2, lambda3, lambda4, rho2, rho3, mu3, \
    # mu4, mu6, upsilon1, upsilon2, upsilon3, upsilon4, omega1, omega2, \
    # omega3, A_MII0, I0, beta0, A_MC0, A_F0, A_M0 = parameters

    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, \
    gamma, zeta, f_dillution, lambda1, lambda2, lambda3, lambda4, \
    rho1, rho2, rho3, mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8, \
    upsilon1, upsilon2, upsilon3, upsilon4, omega1, omega2, omega3, \
    A_MII0, I0, beta0, A_MC0, A_F0, A_M0, A_Malpha0, CIII0, CI0 = params
 
    day_conversion = 24 * 60
    
    k1 = 2.34 * 10**-5
    k2 =  234 * 10**-5 * day_conversion
    k4 = 280 * 10**-5 * day_conversion
    k7 =  50 
    k8 = 30 
    k11 = 2 * 10**(-7) * day_conversion
    lambda1 = 0.001 * day_conversion
    rho1 =  0.3
    mu1 = 0.07
    mu2 = 7
    mu5 = 0.1
    mu7 = 9.7 * 10**(-5) * day_conversion
    mu8 = 9.7 * 10**(-5) * day_conversion
    gamma = 10**(-5)
    zeta = 10**(-5)
    A_Malpha0 = 0
    CIII0 = 0
    CI0 = 0
    f_dillution = 1/16

    parameters = {
        'k1': k1, 'k2': k2, 'k3': k3, 'k4': k4, 'k5': k5, 'k6': k6, 'k7': k7, 'k8': k8, 'k9': k9, 'k10': k10, 'k11': k11,
        'gamma': gamma,'zeta': zeta, 'f_dillution': f_dillution,
        'lambda1': lambda1, 'lambda2': lambda2, 'lambda3': lambda3, 'lambda4': lambda4,
        'rho1': rho1, 'rho2': rho2, 'rho3': rho3,
        'mu1': mu1, 'mu2': mu2, 'mu3': mu3, 'mu4': mu4, 'mu5': mu5, 'mu6': mu6, 'mu7': mu7, 'mu8': mu8,
        'upsilon1': upsilon1, 'upsilon2': upsilon2, 'upsilon3': upsilon3, 'upsilon4': upsilon4,
        'omega1': omega1, 'omega2': omega2, 'omega3': omega3,
        'A_MII0': A_MII0, 'I0': I0, 'beta0': beta0, 'A_MC0': A_MC0, 'A_F0': A_F0, 'A_M0': A_M0, 'A_Malpha0': A_Malpha0,
        'CIII0': CIII0, 'CI0': CI0
    }
    # Time parameters
    weeks = 30
    n_days_in_week = 7
    t_max = weeks * n_days_in_week  # Maximum simulation time (weeks)
    dt = weeks / t_max  # Time step

    # Forward Euler method
    timesteps = int(t_max / dt)
    time = np.linspace(0, t_max, timesteps)

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

    A_MII2 = [A_MII0]
    I2 = [I0]
    beta2 = [beta0]
    A_MC2 = [A_MC0]
    A_F2 = [A_F0]
    A_M2 = [A_M0]
    A_Malpha2 = [A_Malpha0]
    CIII2 = [CIII0]
    CI2 = [CI0]

    # Perform simulation for both scenarios using forward Euler method
    for i in range(1, timesteps + 1):
        t = i * dt

        # Scenario 1
        A_MII_next, I_next, beta_next, A_MC_next, A_F_next, A_M_next, A_Malpha_next, CIII_next, CI_next = \
            scenario1_equations(A_MII1[-1], I1[-1], beta1[-1], A_MC1[-1], A_F1[-1], A_M1[-1], A_Malpha1[-1], CIII1[-1], CI1[-1], parameters, dt, t)
        A_MII1 = np.append(A_MII1, A_MII_next)
        I1 = np.append(I1, I_next)
        beta1 = np.append(beta1, beta_next)
        A_MC1 = np.append(A_MC1, A_MC_next)
        A_F1 = np.append(A_F1, A_F_next)
        A_M1 = np.append(A_M1, A_M_next)
        A_Malpha1 = np.append(A_Malpha1, A_Malpha_next)
        CIII1 = np.append(CIII1, CIII_next)
        CI1 = np.append(CI1, CI_next)

        # Scenario 2
        A_MII_next, I_next, beta_next, A_MC_next, A_F_next, A_M_next, A_Malpha_next, CIII_next, CI_next = \
            scenario2_equations(A_MII2[-1], I2[-1], beta2[-1], A_MC2[-1], A_F2[-1], A_M2[-1], A_Malpha2[-1], CIII2[-1], CI2[-1], parameters, dt, t)
        A_MII2 = np.append(A_MII2, A_MII_next)
        I2 = np.append(I2, I_next)
        beta2 = np.append(beta2, beta_next)
        A_MC2 = np.append(A_MC2, A_MC_next)
        A_F2 = np.append(A_F2, A_F_next)
        A_M2 = np.append(A_M2, A_M_next)
        A_Malpha2 = np.append(A_Malpha2, A_Malpha_next)
        CIII2 = np.append(CIII2, CIII_next)
        CI2 = np.append(CI2, CI_next)

    return A_MII1, A_MC1, A_F1



# Define the quadratic curve function
def quadratic_curve(t, a, b, c):
    return a * t ** 2 + b * t + c

# Function to fit quadratic curves to the given data
def fit_quadratic_curves(data, A0, time_weeks):

    # Convert time_weeks and data to float64 dtype
    time_weeks = np.array(time_weeks, dtype=np.float64)
    data = np.array(data, dtype=np.float64)
    A0 = float(A0)
    popt, _ = curve_fit(quadratic_curve, time_weeks, data * A0)
    return quadratic_curve(time_weeks, *popt)

def cell_dynamics_curve(time, parameters):
    # values = list(parameters.values())
    if callable(parameters):
        temploc = parameters().values()
    elif isinstance(parameters, np.ndarray):
        temploc = parameters
    else:
        temploc = parameters.values()
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11,\
    gamma, zeta, f_dillution, lambda1, lambda2, lambda3, lambda4,\
    rho1, rho2, rho3, mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8,\
    upsilon1, upsilon2, upsilon3, upsilon4, omega1, omega2, omega3,\
    A_MII0, I0, beta0, A_MC0, A_F0, A_M0, A_Malpha0, CIII0, CI0 = temploc

    # Quantities for each cell type
    A_MC = np.array([1.0, 0.8, 0.6, 0.3])  # Mast Cells data points
    A_F = np.array([0.5, 0.8, 0.5, 0.4])  # Fibroblast data points
    A_MII = np.array([1.0, 0.8, 0.6, 0.4])  # MII Macrophages data points

    # Extend arrays to match the length of time
    extended_time = np.linspace(0, time[-1], len(time))
    original_time = np.linspace(0, time[-1], len(A_MC))  # Original time points

    # Interpolate the original data points to match the length of time
    A_MC_extended = np.interp(extended_time, original_time, A_MC)
    A_F_extended = np.interp(extended_time, original_time, A_F)
    A_MII_extended = np.interp(extended_time, original_time, A_MII)
    
    # Fit quadratic curves to the given data
    A_MII1 = fit_quadratic_curves(A_MII_extended, A_MII0, extended_time)
    A_MC1 = fit_quadratic_curves(A_MC_extended, A_MC0, extended_time)
    A_F1 = fit_quadratic_curves(A_F_extended, A_F0, extended_time)
    
    return A_MII1, A_MC1, A_F1


# Define the cost function (RMSE) to minimize
def cost_function(parameters, *args):
    
    target_time, target_A_MII, target_A_MC, target_A_F = args
    A_MII, A_MC, A_F = run_simulation(parameters)
    fitted_A_MII, fitted_A_MC, fitted_A_F = cell_dynamics_curve(target_time, parameters)
    rmse_A_MII = np.sqrt(np.mean((fitted_A_MII - A_MII[:-1]) ** 2))
    rmse_A_MC = np.sqrt(np.mean((fitted_A_MC - A_MC[:-1]) ** 2))
    rmse_A_F = np.sqrt(np.mean((fitted_A_F - A_F[:-1]) ** 2))
    
    rmse = rmse_A_MII + rmse_A_MC + rmse_A_F  # Combined RMSE for all cell types
    return rmse

# Time parameters
weeks = 30
n_days_in_week = 7
t_max =  weeks * n_days_in_week # Maximum simulation time(weeks)
dt = weeks/t_max # Time step

# Forward Euler method
timesteps = int(t_max / dt)
time = np.linspace(0, t_max, timesteps)
def_params = defined_params()  # Parameters to ignore and never change
params = initial_parameters() 
target_A_MII, target_A_MC, target_A_F = cell_dynamics_curve(time, params)

# print(initial_params['f_dillution'])
# print(initial_params['k1'])
# print(initial_params['mu1'])


# Optimize the parameters
result_list = []  # To store the best 10 results
everchanaging_params = params.copy()
for _ in range(100):  # Perform optimization 10 times and store the best results
    # Get initial parameter values
    # Exclude parameters specified in def_params from optimization
    for param in def_params.keys():
            everchanaging_params[param] = def_params[param]
    # print(rest)
    # Define parameter bounds using the ranges from parameter_ranges dictionary
    bounds = []

    for param in everchanaging_params.keys():
        if param in p_ranges:
            # If the parameter is in parameter_ranges, use the specified range
            param_min, param_max = float(p_ranges[param][0]), float(p_ranges[param][1])
        else:
            # If the parameter is not in parameter_ranges, use default bounds (0, 1)
            param_min, param_max = 0, 1
        bounds.append((param_min, param_max))
    # print('bounds',np.shape(bounds))
    # print('initial_params',np.shape(initial_params))
    # print('time',np.shape(time))
    # print('target_A_MII',np.shape(target_A_MII))
    # print('target_A_MC',np.shape(target_A_MC))
    # print('target_A_F',np.shape(target_A_F))
    # print(initial_params.values())
    
    # Perform optimization
    # print(initial_params.keys())
    result = minimize(cost_function, list(everchanaging_params.values()), args=(time, target_A_MII, target_A_MC, target_A_F),
                  bounds=bounds)
    
    # Reconstruct the full parameter dictionary including parameters from def_params
    optimized_params = params.copy()
    for i, param in enumerate(everchanaging_params.keys()):
        optimized_params[param] = result.x[i]
    
    # Store the result
    result_dict = {
        'parameters': optimized_params,
        'error': result.fun
    }
    result_list.append(result_dict)


# Sort the results by error and keep the top 10
result_list = sorted(result_list, key=lambda x: x['error'])[:10]

# Save the best 10 results to a JSON file
with open('best_results.json', 'w') as json_file:
    json.dump(result_list, json_file)

print('Optimization complete. Best results saved to best_results.json.')