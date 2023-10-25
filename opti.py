import numpy as np
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import copy
from params import initial_parameters
# from main import A_MII1_func,A_MII2_func,A_Malpha_func, scenario1_equations, scenario2_equations




def A_MII1_func(mu1, A_MII0, t):
    A_MII1 = np.exp(-mu1 * t) * A_MII0
    return A_MII1 

# a = A_MII1_func(k1, mu1, A_MII0, time)

def A_MII2_func(k1, mu1, A_MII0, omega1, t):
    A_MII2 = k1 * np.exp(-mu1 * t) * np.cos(omega1 * t) * A_MII0
    return A_MII2 

def A_Malpha_func(rho2, A_M, A_Malpha0, t):
    A_Malpha = rho2 * A_M * t * A_Malpha0
    return A_Malpha


# Scenario 1 equations
def scenario1_equations(A_MII, I, beta, A_MC, A_F, A_M, A_Malpha, CIII, CI, parameters, dt, t):
    A_MII_next = parameters['f_dillution'] * parameters['k1'] * np.exp(-parameters['mu1'] * t) 
    I_next = I + dt  * parameters['f_dillution'] * (-parameters['k2'] * parameters['upsilon1'] * np.exp(-parameters['upsilon1'] * t) + parameters['k6'] * parameters['gamma'] * A_MC - parameters['mu2'] * I)
    beta_next = beta + dt  * parameters['f_dillution'] * (parameters['k3'] * parameters['upsilon2'] * np.exp(parameters['upsilon2'] * t) + parameters['k4'] * parameters['gamma'] * A_MII + parameters['k5'] * parameters['gamma'] * A_MC - parameters['mu3'] * beta)
    A_MC_next = A_MC + dt * (A_MC * I * parameters['lambda3'] * parameters['zeta'] - parameters['mu4'] * A_MC)
    A_F_next = A_F + dt * (parameters['lambda2'] * parameters['zeta'] * I * A_F - parameters['lambda1'] * parameters['zeta'] * beta * A_F - parameters['rho1'] * A_F - parameters['mu5'] * A_F)
    A_M_next = A_M + dt * (parameters['rho1'] * A_F + parameters['lambda1'] * parameters['zeta'] * beta * A_F + parameters['lambda4'] * parameters['zeta'] * A_F * A_M - parameters['mu6'] * A_M)
    A_Malpha_next = A_Malpha_func(parameters['rho2'], A_M, parameters['A_Malpha0'], t)
    CIII_next = CIII + dt * (parameters['k7'] * parameters['gamma'] * A_F + parameters['k10'] * parameters['gamma'] * A_M - parameters['rho3'] * CIII - parameters['mu7'] * CIII)
    CI_next = CI + dt * (parameters['rho3'] * CIII + parameters['k9'] * parameters['gamma'] * A_M + parameters['k4'] * parameters['gamma'] * A_Malpha + parameters['k8'] * parameters['gamma'] * A_F * parameters['k11'] * parameters['gamma'] * A_Malpha - parameters['mu8'] * CI)
    return A_MII_next, I_next, beta_next, A_MC_next, A_F_next, A_M_next, A_Malpha_next, CIII_next, CI_next

# Scenario 2 equations
def scenario2_equations(A_MII, I, beta, A_MC, A_F, A_M, A_Malpha, CIII, CI, parameters, dt, t):
    A_MII_next = parameters['f_dillution'] * parameters['k1']*  np.exp(-parameters['mu1'] * t) * np.cos(parameters['omega1'] * t) 
    I_next = I + dt  * parameters['f_dillution'] * (-parameters['k2'] * parameters['upsilon3'] * np.exp(-parameters['upsilon3'] * t) * np.cos(parameters['omega2'] * t)
                       - parameters['k2']  * np.exp(-parameters['upsilon3'] * t) * parameters['omega2'] * np.sin(parameters['omega2'] * t) + parameters['k6'] * parameters['gamma'] * A_MC - parameters['mu2'] * I)
    beta_next = beta + dt  * parameters['f_dillution'] * (parameters['k3'] * np.exp(-parameters['upsilon4'] * t) * parameters['omega3'] * np.cos(parameters['omega3'] * t)
                             - parameters['k3']  * parameters['upsilon4'] * np.exp(-parameters['upsilon4'] * t) * np.sin(parameters['omega3'] * t)
                             + parameters['k4'] * parameters['gamma'] * A_MII + parameters['k5'] * parameters['gamma'] * A_MC - parameters['mu3'] * beta)
    A_MC_next = A_MC + dt * (A_MC * I * parameters['lambda3'] * parameters['zeta'] - parameters['mu4'] * A_MC)
    A_F_next = A_F + dt * (parameters['lambda2'] * parameters['zeta'] * I * A_F - parameters['lambda1'] * parameters['zeta'] * beta * A_F - parameters['rho1'] * A_F - parameters['mu5'] * A_F)
    A_M_next = A_M + dt * (parameters['rho1'] * A_F + parameters['lambda1'] * parameters['zeta'] * beta * A_F + parameters['lambda4'] * parameters['zeta'] * A_F * A_M - parameters['mu6'] * A_M)
    A_Malpha_next = A_Malpha_func(parameters['rho2'], A_M, parameters['A_Malpha0'], t)
    CIII_next = CIII + dt * (parameters['k7'] * parameters['gamma'] * A_F + parameters['k10'] * parameters['gamma'] * A_M - parameters['rho3'] * CIII - parameters['mu7'] * CIII)
    CI_next = CI + dt * (parameters['rho3'] * CIII + parameters['k9'] * parameters['gamma'] * A_M + parameters['k4'] * parameters['gamma'] * A_Malpha + parameters['k8'] * parameters['gamma'] * A_F * parameters['k11'] * parameters['gamma'] * A_Malpha - parameters['mu8'] * CI)
    return A_MII_next, I_next, beta_next, A_MC_next, A_F_next, A_M_next, A_Malpha_next, CIII_next, CI_next


day_conversion = 60 * 24

ranges_for_production = (0.001, 10)
ranges_for_upsi = (0.001, 0.1)
ranges_for_omega = (0, 2 * np.pi)
ranges_for_gamma = (1.0 * 10**-7, 1.0 * 10**-5)
ranges_for_zeta = (1.0 * 10**-7, 1.0 * 10**-1)
ranges_for_mu = (0.0001, 10)
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
  'A_MII0': (0, 3000),
  'I0': (10**-9, 10**-6),
  'beta0': (10**-9, 10**-6),
  'A_MC0': (0, 3000),
  'A_F0': (0, 3000),
  'A_M0': (0, 3000),
  'A_Malpha0':(0, 0.000001),
   'CI0':(0, 0.000001),
   'CIII0':(0, 0.000001), 
   'omega1': ranges_for_omega, 
   'omega2': ranges_for_omega, 
   'omega3': ranges_for_omega,
   'gamma': (10**(-5), 10**(-4)),
    'zeta': (10**(-5), 10**(-4)),
    'f_dillution': (1/16, 1/15),
}


# Define your simulation function here (using the initial_parameters)
def run_simulation(parameters, scenario_nr):
    # Time parameters
    weeks = 30
    n_days_in_week = 7
    t_max =  weeks * n_days_in_week # Maximum simulation time(weeks)
    dt = weeks/t_max # Time step

    # Forward Euler method
    timesteps = int(t_max / dt)
    time = np.linspace(0, t_max, timesteps)


    # Initialize arrays for results
    A_MII1 = [parameters['A_MII0']]
    I1 = [parameters['I0']]
    beta1 = [parameters['beta0']]
    A_MC1 = [parameters['A_MC0']]
    A_F1 = [parameters['A_F0']]
    A_M1 = [parameters['A_M0']]
    A_Malpha1 = [parameters['A_Malpha0']]
    CIII1 = [parameters['CIII0']]
    CI1 = [parameters['CI0']]

    # Initialize arrays for results
    A_MII2 = [parameters['A_MII0']]
    I2 = [parameters['I0']]
    beta2 = [parameters['beta0']]
    A_MC2 = [parameters['A_MC0']]
    A_F2 = [parameters['A_F0']]
    A_M2 = [parameters['A_M0']]
    A_Malpha2 = [parameters['A_Malpha0']]
    CIII2 = [parameters['CIII0']]
    CI2 = [parameters['CI0']]

    # Perform simulation for both scenarios using forward Euler method
    for i in range(1, timesteps + 1):
        t = i * dt
        if scenario_nr == "1":
          # Scenario 1
          A_MII_next, I_next, beta_next, A_MC_next, A_F_next, A_M_next, A_Malpha_next, CIII_next, CI_next = \
              scenario1_equations(A_MII1[-1], I1[-1], beta1[-1], A_MC1[-1], A_F1[-1], A_M1[-1], A_Malpha1[-1], CIII1[-1], CI1[-1], parameters, dt, t)
          A_MII1.append(A_MII_next)
          I1.append(I_next)
          beta1.append(beta_next)
          A_MC1.append(A_MC_next)
          A_F1.append(A_F_next)
          A_M1.append(A_M_next)
          A_Malpha1.append(A_Malpha_next)
          CIII1.append(CIII_next)
          CI1.append(CI_next)
        if scenario_nr == "2":
          # Scenario 2
          A_MII_next, I_next, beta_next, A_MC_next, A_F_next, A_M_next, A_Malpha_next, CIII_next, CI_next = \
              scenario2_equations(A_MII2[-1], I2[-1], beta2[-1], A_MC2[-1], A_F2[-1], A_M2[-1], A_Malpha2[-1], CIII2[-1], CI2[-1], parameters, dt, t)
          A_MII2.append(A_MII_next)
          I2.append(I_next)
          beta2.append(beta_next)
          A_MC2.append(A_MC_next)
          A_F2.append(A_F_next)
          A_M2.append(A_M_next)
          A_Malpha2.append(A_Malpha_next)
          CIII2.append(CIII_next)
          CI2.append(CI_next)

    if scenario_nr == "1":
      output = A_MII1[1:], A_MC1[1:], A_F1[1:]
    if scenario_nr == "2":
      output = A_MII2[1:], A_MC2[1:], A_F2[1:]

    return time, output
# Define the quadratic curve function
def quadratic_curve(t, a, b, c):
    return a * t ** 2 + b * t + c +  1e-10

# Function to fit quadratic curves to the given data
def fit_quadratic_curves(data, A0, time):
    data = np.array(data, dtype=np.float64)
    # Replace NaN and infinite values with 0
    cleaned_data = np.nan_to_num(data, nan=0, posinf=0, neginf=0)
    # print(data)
    A0 = float(A0)
    popt, _ = curve_fit(quadratic_curve, time, cleaned_data * A0)
    return quadratic_curve(time, *popt)

# Define a function to generate initial parameters with added noise within specified ranges
def generate_noisy_parameters(initial_guess, p_ranges, noise_level):
    noisy_params = copy.deepcopy(initial_guess)  # Create a deep copy of the initial guess
    for param in noisy_params:
        # Add noise to the parameter in both positive and negative directions
        noisy_param = noisy_params[param] + np.random.uniform(-noise_level, noise_level)
        # Ensure the noisy parameter stays within the specified range
        min_val, max_val = p_ranges[param]
        # Clip noisy_param to the valid range defined by min_val and max_val
        noisy_param = np.clip(noisy_param, min_val, max_val)
        noisy_params[param] = noisy_param
    return noisy_params


# Define the objective (cost) function to minimize RMSE
def objective_function(params, scenario_nr):
    # print(params)
    # Extract parameters
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, \
    gamma, zeta, f_dillution, lambda1, lambda2, lambda3,\
     lambda4, rho1, rho2, rho3, mu1, mu2, mu3, mu4, mu5, \
     mu6, mu7, mu8, upsilon1, upsilon2, upsilon3, upsilon4, \
     omega1, omega2, omega3, A_MII0, I0, beta0, A_MC0, A_F0, \
     A_M0, A_Malpha0, CIII0, CI0 = params  # Extract parameters individually
    print(params)
    # A_MII0, A_MC0, A_F0 = 2000, 1000, 500
    # Run simulation with current parameters
    res = run_simulation(parameters = {
        'k1': k1, 'k2': k2, 'k3': k3, 'k4': k4, 'k5': k5, 'k6': k6, 'k7': k7, 'k8': k8, 'k9': k9, 'k10': k10, 'k11': k11,
        'gamma': gamma,'zeta': zeta, 'f_dillution': f_dillution,
        'lambda1': lambda1, 'lambda2': lambda2, 'lambda3': lambda3, 'lambda4': lambda4,
        'rho1': rho1, 'rho2': rho2, 'rho3': rho3,
        'mu1': mu1, 'mu2': mu2, 'mu3': mu3, 'mu4': mu4, 'mu5': mu5, 'mu6': mu6, 'mu7': mu7, 'mu8': mu8,
        'upsilon1': upsilon1, 'upsilon2': upsilon2, 'upsilon3': upsilon3, 'upsilon4': upsilon4,
        'omega1': omega1, 'omega2': omega2, 'omega3': omega3,
        'A_MII0': A_MII0, 'I0': I0, 'beta0': beta0, 'A_MC0': A_MC0, 'A_F0': A_F0, 'A_M0': A_M0, 'A_Malpha0': A_Malpha0,
        'CIII0': CIII0, 'CI0': CI0}, scenario_nr = scenario_nr)
    time = res[0] 



    fitted_A_MII1, fitted_A_MC1, fitted_A_F1 = res[1]

    # Quantities for each cell type
    A_MC = np.array([1.0, 0.8, 0.6, 0.3])  # Mast Cells data points
    A_F = np.array([0.5, 0.8, 0.5, 0.4])  # Fibroblast data points
    A_MII = np.array([1.0, 0.8, 0.6, 0.4])  # MII Macrophages data points

    extended_time = np.linspace(0, time[-1], len(time))
    original_time = np.linspace(0, time[-1], len(A_MC))


    # Interpolate the original data points to match the length of time
    A_MC_extended = np.interp(extended_time, original_time, A_MC)
    A_F_extended = np.interp(extended_time, original_time, A_F)
    A_MII_extended = np.interp(extended_time, original_time, A_MII)

    
    # Fit quadratic curves to simulation results
    target_A_MII1 = fit_quadratic_curves(A_MII_extended, A_MII0, extended_time)
    target_A_MC1 = fit_quadratic_curves(A_MC_extended, A_MC0, extended_time)
    target_A_F1 = fit_quadratic_curves(A_F_extended, A_F0, extended_time)
    # print(target_A_F1)
    # Replace NaN and infinite values with 0
    fitted_A_MII1 = np.nan_to_num(fitted_A_MII1, nan=0, posinf=0, neginf=0)
    # Replace NaN and infinite values with 0
    fitted_A_MC1 = np.nan_to_num(fitted_A_MC1, nan=0, posinf=0, neginf=0)
    # Replace NaN and infinite values with 0
    fitted_A_F1 = np.nan_to_num(fitted_A_F1, nan=0, posinf=0, neginf=0)


    # Calculate RMSE for each curve and combine them into a single objective value
    rmse_A_MII1 = np.sqrt(np.mean((fitted_A_MII1 - target_A_MII1)**2))
    rmse_A_MC1 = np.sqrt(np.mean((fitted_A_MC1 - target_A_MC1)**2))
    rmse_A_F1 = np.sqrt(np.mean((fitted_A_F1 - target_A_F1)**2))
    
    # Objective function to minimize RMSE
    objective = rmse_A_MII1 + rmse_A_MC1 + rmse_A_F1
    print(objective)
    return objective


day_conversion = 60 * 24 #
# Production Parameters 
k1 = float(2.34 * 10**-6) # rho2 https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009839
k2 = 234 * 10**-5 * day_conversion # day combi model
k3 = 0.15
k4 = 280 * 10**-5 * day_conversion # day combi model
k5 = 0.2
k6 = 0.3 
k7 = 50 #* 10**-5 # k1 and k2 between 30-60 https://zero.sci-hub.se/3771/b68709ea5f5640da4199e36ff25ef036/cumming2009.pdf
k8 = 30 #* 10**-5 # k1 and k2 between 30-60 https://zero.sci-hub.se/3771/b68709ea5f5640da4199e36ff25ef036/cumming2009.pdf
k9 = 0.23
k10 = 0.1
k11 = 2 * 10**(-7) * day_conversion # https://www.sciencedirect.com/science/article/pii/S0045782516302857?casa_token=ByHEzHgojSEAAAAA:XNdfPARqEPtiO3rcqb0jo9d--utWdu-swPxNKOLyK5huphzY4TcRxiVo4c4yzCASMY-tOswVpzY#br000425

# Conversion parameters
gamma = 10**(-5)
zeta = 10**(-5)
f_dillution = float(1/16) #

# Activation parameters
lambda1 = 0.001 * day_conversion # https://www.sciencedirect.com/science/article/pii/S0045782516302857?casa_token=ByHEzHgojSEAAAAA:XNdfPARqEPtiO3rcqb0jo9d--utWdu-swPxNKOLyK5huphzY4TcRxiVo4c4yzCASMY-tOswVpzY#br000435
lambda2 = 0.04
lambda3 = 0.08
lambda4 = 0.03

# Transition parameters
rho1 = 0.3 # https://link.springer.com/article/10.1007/s11538-012-9751-z/tables/1
rho2 = 0.02
rho3 = 0.01

# Decay parameters
mu1 = float(0.07) # day-1 mu_AM https://link.springer.com/article/10.1186/1471-2105-14-S6-S7/tables/2
mu2 = 7 # day-1 mu_CH https://link.springer.com/article/10.1186/1471-2105-14-S6-S7/tables/2
mu3 = 0.03
mu4 = 0.01
mu5 = 0.1 # https://link.springer.com/article/10.1007/s11538-012-9751-z/tables/1
mu6 = 0.04
mu6 = 0.03
mu7 = 9.7 * 10**(-5) * day_conversion # https://www.sciencedirect.com/science/article/pii/S0045782516302857?casa_token=ByHEzHgojSEAAAAA:XNdfPARqEPtiO3rcqb0jo9d--utWdu-swPxNKOLyK5huphzY4TcRxiVo4c4yzCASMY-tOswVpzY#br000475
mu8 = 9.7 * 10**(-5) * day_conversion # https://www.sciencedirect.com/science/article/pii/S0045782516302857?casa_token=ByHEzHgojSEAAAAA:XNdfPARqEPtiO3rcqb0jo9d--utWdu-swPxNKOLyK5huphzY4TcRxiVo4c4yzCASMY-tOswVpzY#br000475

# sinusoidal parameters
upsilon1 = -0.02 # negative value
upsilon2 = 0.03
upsilon3 = 0.01
upsilon4 = 0.02
omega1 = 0.5
omega2 = 0.7
omega3 = 0.6

# Initial conditions
A_MII0 = 2000
I0 = 10**(-9) #
beta0 = 10**(-7) #
A_MC0 = 100
A_F0 = 500
A_M0 = 20
A_Malpha0 = 0
CIII0 = 0
CI0 = 0

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



initial_guess = parameters
# Set the noise level (adjust this value based on the desired sensitivity)
noise_level = 0.001  # Example noise level, you can adjust this
# Generate noisy initial parameters with added noise within specified ranges
noisy_initial_params = generate_noisy_parameters(initial_guess, p_ranges, noise_level)
scenario_nr = "1"

weeks = 30
n_days_in_week = 7
t_max =  weeks * n_days_in_week # Maximum simulation time(weeks)
dt = 0.1 # Time step

# Forward Euler method
timesteps = int(t_max / dt)
time = np.linspace(0, t_max, timesteps)


# print(len(initial_params))
# Perform optimization with bounds as specified in p_ranges
result = minimize(objective_function, list([  1.44, 3.3728033538319924, 1.44, 4.023532898817432, 1.44, 1.44, 5.0, 5.0, 1.44, 1.44, 1.44, 0.0, 0.0001, 1.44, 1.44, 0.14400000000000002, 0.14400000000000002, 0.14400000000000002, 0.29983680419899583, 0.14400000000000002, 0.14400000000000002, 0.14400000000000002, 1.0, 0.14400000000000002, 0.14400000000000002, 0.14400000000000002, 0.14400000000000002, 0.14400000000000002, 0.14400000000000002, 0.14400000000000002, 1000.0, 1e-06, 1e-06, 0.5000409314065507, 0.6932532259617087, 0.6096239485272107, 1e-06, 1e-06, 1e-09, 6.283185307179586, 6.283185307179586, 6.283185307179586, 1e-05, 1e-05, 0.0625
  ]), args = (scenario_nr), method='L-BFGS-B', 
                  bounds=[(p_ranges[param][0], p_ranges[param][1]) for param in p_ranges.keys()])

# Extract optimized parameters
optimized_params = result.x
# Print optimized parameters with a comma between each one of them
print("Optimized Parameters: " + ", ".join(map(str, optimized_params)))

