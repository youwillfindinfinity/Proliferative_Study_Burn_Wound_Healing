import numpy as np
from SALib.sample import saltelli, sobol
# from SALib.analyze import sobol
import json



def A_MII1_func(k1, mu1, A_MII0, t):
    A_MII1 = np.exp(-mu1 * t) * A_MII0
    return A_MII1 

# a = A_MII1_func(k1, mu1, A_MII0, time)

def A_MII2_func(k1, mu1, A_MII0, omega1, t):

    A_MII2 = np.exp(-mu1 * t) * np.cos(omega1 * t) * A_MII0
    return A_MII2 

def A_Malpha_func(rho2, A_M, A_Malpha0, t):
    A_Malpha = rho2 * A_M * t * A_Malpha0
    return A_Malpha


# Scenario 1 equations
def scenario1_equations(A_MII, I, beta, A_MC, A_F, A_M, A_Malpha, CIII, CI, parameters, dt, t):
    A_MII_next = A_MII1_func(parameters['k1'], parameters['mu1'], parameters['A_MII0'], t)
    I_next = I + dt  * (-parameters['k2'] * parameters['upsilon1'] * np.exp(-parameters['upsilon1'] * t) + parameters['k6'] * parameters['gamma'] * A_MC - parameters['mu2'] * I)
    beta_next = beta + dt * (parameters['k3'] * parameters['upsilon2'] * np.exp(parameters['upsilon2'] * t) + parameters['k4'] * parameters['gamma'] * A_MII + parameters['k5'] * parameters['gamma'] * A_MC - parameters['mu3'] * beta)
    A_MC_next = A_MC + dt * (A_MC * I * parameters['lambda3'] * parameters['zeta'] - parameters['mu4'] * A_MC)
    A_F_next = A_F + dt * (parameters['lambda2'] * parameters['zeta'] * I * A_F + parameters['lambda1'] * parameters['zeta'] * beta * A_F - parameters['rho1'] * A_F - parameters['mu5'] * A_F)
    A_M_next = A_M + dt * (parameters['rho1'] * A_F + parameters['lambda1'] * parameters['zeta'] * beta * A_F + parameters['lambda4'] * parameters['zeta'] * A_F * A_M - parameters['mu6'] * A_M)
    A_Malpha_next = A_Malpha_func(parameters['rho2'], A_M, parameters['A_Malpha0'], t)
    CIII_next = CIII + dt * (parameters['k7'] * parameters['gamma'] * A_F + parameters['k10'] * parameters['gamma'] * A_M - parameters['rho3'] * CIII - parameters['mu7'] * CIII)
    CI_next = CI + dt * (parameters['rho3'] * CIII + parameters['k9'] * parameters['gamma'] * A_M + parameters['k4'] * parameters['gamma'] * A_Malpha + parameters['k8'] * parameters['gamma'] * A_F * parameters['k11'] * parameters['gamma'] * A_Malpha - parameters['mu8'] * CI)
    return A_MII_next, I_next, beta_next, A_MC_next, A_F_next, A_M_next, A_Malpha_next, CIII_next, CI_next

# Scenario 2 equations
def scenario2_equations(A_MII, I, beta, A_MC, A_F, A_M, A_Malpha, CIII, CI, parameters, dt, t):
    A_MII_next = A_MII2_func(parameters['k1'], parameters['mu1'], parameters['A_MII0'], parameters['omega1'], t)
    I_next = I + dt  * (-parameters['k2'] * parameters['upsilon3'] * np.exp(-parameters['upsilon3'] * t) * np.cos(parameters['omega2'] * t)
                       - parameters['k2'] * np.exp(-parameters['upsilon3'] * t) * parameters['omega2'] * np.sin(parameters['omega2'] * t) + parameters['k6'] * parameters['gamma'] * A_MC- parameters['mu2'] * I)
    beta_next = beta + dt  * (parameters['k3'] * np.exp(-parameters['upsilon4'] * t) * parameters['omega3'] * np.cos(parameters['omega3'] * t)
                             - parameters['k3'] * parameters['upsilon4'] * np.exp(-parameters['upsilon4'] * t) * np.sin(parameters['omega3'] * t)
                             + parameters['k4'] * parameters['gamma'] * A_MII + parameters['k5'] * parameters['gamma'] * A_MC - parameters['mu3'] * beta)
    A_MC_next = A_MC + dt * (A_MC * I * parameters['lambda3'] * parameters['zeta'] - parameters['mu4'] * A_MC)
    A_F_next = A_F + dt * (parameters['lambda2'] * parameters['zeta'] * I * A_F - parameters['lambda1'] * parameters['zeta'] * beta * A_F - parameters['rho1'] * A_F - parameters['mu5'] * A_F)
    A_M_next = A_M + dt * (parameters['rho1'] * A_F + parameters['lambda1'] * parameters['zeta'] * beta * A_F + parameters['lambda4'] * parameters['zeta'] * A_F * A_M - parameters['mu6'] * A_M)
    A_Malpha_next = A_Malpha_func(parameters['rho2'], A_M, parameters['A_Malpha0'], t)
    CIII_next = CIII + dt * (parameters['k7'] * parameters['gamma'] * A_F + parameters['k10'] * parameters['gamma'] * A_M - parameters['rho3'] * CIII - parameters['mu7'] * CIII)
    CI_next = CI + dt * (parameters['rho3'] * CIII + parameters['k9'] * parameters['gamma'] * A_M + parameters['k4'] * parameters['gamma'] * A_Malpha + parameters['k8'] * parameters['gamma'] * A_F * parameters['k11'] * parameters['gamma'] * A_Malpha - parameters['mu8'] * CI)
    return A_MII_next, I_next, beta_next, A_MC_next, A_F_next, A_M_next, A_Malpha_next, CIII_next, CI_next

dt = 1/(60*24)
dt_hour = 1/60
# Production Parameters 
k1 = 2.34 * 10**(-5) * dt #2.34 * 10**(-5) * dt *rho2 https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009839
k2 = 2.34 * 10**(-5) * dt #2.34 * 10**(-5) * dt  combi model ****
k3 = 1 * 10**(-5) * dt # 1 * 10**(-5) * dt 
k4 = 8.80 * 10**(-5) * dt # 28.0 * 10**(-5) * dt day combi model *****
k5 = 0.00001 * dt # 0.00001 * dt
k6 = 0.0005 * dt # 0.0005 * dt
k7 = 30 * dt #30 * dt  k1 and k2 between 30-60 https://zero.sci-hub.se/3771/b68709ea5f5640da4199e36ff25ef036/cumming2009.pdf
k8 = 1 * dt # 1 * dt https://link.springer.com/article/10.1007/s11538-012-9751-z/tables/1
k9 = 50 * dt # 50 * dt
k10 = 30 * dt # 30 * dt
k11 = 20 * dt # 2000000 * dt production of CI forced by myofibroblasts https://www.sciencedirect.com/science/article/pii/S0045782516302857?casa_token=ByHEzHgojSEAAAAA:XNdfPARqEPtiO3rcqb0jo9d--utWdu-swPxNKOLyK5huphzY4TcRxiVo4c4yzCASMY-tOswVpzY#br000425

# Conversion parameters
gamma = 10**(-5) 
zeta = 10**(5) # fixed
f_dillution = 1/16 

# Activation parameters
lambda1 = 100 * dt # 100 * dt https://www.sciencedirect.com/science/article/pii/S0045782516302857?casa_token=ByHEzHgojSEAAAAA:XNdfPARqEPtiO3rcqb0jo9d--utWdu-swPxNKOLyK5huphzY4TcRxiVo4c4yzCASMY-tOswVpzY#br000435
lambda2 = 400 * dt # 400 * dt
lambda3 = 300 * dt # 300 * dt
lambda4 = 10**(-7) * dt # 10**(-7) * dt 

# Transition parameters
rho1 = 5 * dt
rho2 = 3 * dt
rho3 = 18 * dt # transition CIII to CI

# # Decay parameters
mu1 = 2 * dt # 2 * dt# day-1 mu_AM https://link.springer.com/article/10.1186/1471-2105-14-S6-S7/tables/2
mu2 = 12 * dt # 12 * dt  day-1 mu_CH https://link.springer.com/article/10.1186/1471-2105-14-S6-S7/tables/2
mu3 = 10 * dt # 10 * dt
mu4 = 11 * dt # 11 * dt
mu5 = 10 * dt # 10 * dt day https://link.springer.com/article/10.1007/s11538-012-9751-z/tables/1
mu6 = 10 * dt # 10 * dt
mu7 = 5 * dt # 9.7 * 10**(-5) * dt https://www.sciencedirect.com/science/article/pii/S0045782516302857?casa_token=ByHEzHgojSEAAAAA:XNdfPARqEPtiO3rcqb0jo9d--utWdu-swPxNKOLyK5huphzY4TcRxiVo4c4yzCASMY-tOswVpzY#br000475
mu8 = 1 * dt # 9.7 * 10**(-5) * dthttps://www.sciencedirect.com/science/article/pii/S0045782516302857?casa_token=ByHEzHgojSEAAAAA:XNdfPARqEPtiO3rcqb0jo9d--utWdu-swPxNKOLyK5huphzY4TcRxiVo4c4yzCASMY-tOswVpzY#br000475

# sinusoidal parameters
upsilon1 = -0.001 #-0.001 # negative value
upsilon2 = -0.1 # -0.1 
upsilon3 = 0.001 # 0.0000001
upsilon4 = 0.001 # 0.0001
omega1 = 1*np.pi *dt # 1*np.pi *dt 
omega2 = 100*np.pi *dt #80*np.pi *dt
omega3 = 80*np.pi *dt #120*np.pi *dt

# Initial conditions
A_MII0 = 1000
I0 = 0.2* 10**(-8) #
beta0 = 2.5 * 10**(-7) #
A_MC0 = 1000
A_F0 = 500
A_M0 = 100
A_Malpha0 = 1
CIII0 = 0.001
CI0 = 0.001







initial_parameters = {
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

# Define a range for incrementing parameters
parameter_increment = 1e-5  # You can adjust this increment value based on your needs

    
# Create the problem dictionary with bounds based on the initial guess and increment
problem = {
    'num_vars': len(initial_parameters),
    'names': list(initial_parameters.keys()),
    'bounds': [[initial_parameters[param] - parameter_increment, initial_parameters[param] + parameter_increment] for param in initial_parameters]
}

# print([[initial_parameters[param] - parameter_increment, initial_parameters[param] + parameter_increment] for param in initial_parameters])
# Check if saved samples exist, if not, generate new samples and save them
try:
    with open('sampled_params.json', 'r') as f:
        param_values_list = json.load(f)
    param_values = np.array(param_values_list)
except FileNotFoundError:
    param_values = saltelli.sample(problem, 1024, calc_second_order=True)
    param_values_list = param_values.tolist()  # Convert NumPy array to list for JSON serialization
    with open('sampled_params.json', 'w') as f:
        json.dump(param_values_list, f)
# Define a function that runs your model with given parameters and returns the output of interest
def model_output(params, scenario=1):
    outputs = []
    for params in param_values:
        # Convert params to a dictionary with parameter names as keys
        sampled_params = {param_name: param_value for param_name, param_value in zip(initial_parameters.keys(), params)}
        
    # Time parameters
    weeks = 30
    n_days_in_week = 7
    t_max =  weeks * n_days_in_week # Maximum simulation time(weeks)
    dt = 1/(60*24)

    # Forward Euler method
    timesteps = int(t_max/dt)
    time = np.linspace(0, t_max, timesteps)
    # print(time, len(time))
    # print(IM)

    # Initialize arrays for results
    A_MII1 = [sampled_params['A_MII0']]
    I1 = [sampled_params['I0']]
    beta1 = [sampled_params['beta0']]
    A_MC1 = [sampled_params['A_MC0']]
    A_F1 = [sampled_params['A_F0']]
    A_M1 = [sampled_params['A_M0']]
    A_Malpha1 = [sampled_params['A_Malpha0']]
    CIII1 = [sampled_params['CIII0']]
    CI1 = [sampled_params['CI0']]

    A_MII2 = [sampled_params['A_MII0']]
    I2 = [sampled_params['I0']]
    beta2 = [sampled_params['beta0']]
    A_MC2 = [sampled_params['A_MC0']]
    A_F2 = [sampled_params['A_F0']]
    A_M2 = [sampled_params['A_M0']]
    A_Malpha2 = [sampled_params['A_Malpha0']]
    CIII2 = [sampled_params['CIII0']]
    CI2 = [sampled_params['CI0']]


    # Perform simulation for both scenarios using forward Euler method
    for i in range(1, timesteps + 1):
        t = i * dt
        
        # Scenario 1
        A_MII_next, I_next, beta_next, A_MC_next, A_F_next, A_M_next, A_Malpha_next, CIII_next, CI_next = \
            scenario1_equations(A_MII1[-1], I1[-1], beta1[-1], A_MC1[-1], A_F1[-1], A_M1[-1], A_Malpha1[-1], CIII1[-1], CI1[-1],sampled_params, dt, t)
        A_MII1.append(A_MII_next)
        I1.append(I_next)
        beta1.append(beta_next)
        A_MC1.append(A_MC_next)
        A_F1.append(A_F_next)
        A_M1.append(A_M_next)
        A_Malpha1.append(A_Malpha_next)
        CIII1.append(CIII_next)
        CI1.append(CI_next)
        
        # Scenario 2
        A_MII_next, I_next, beta_next, A_MC_next, A_F_next, A_M_next, A_Malpha_next, CIII_next, CI_next = \
            scenario2_equations(A_MII2[-1], I2[-1], beta2[-1], A_MC2[-1], A_F2[-1], A_M2[-1], A_Malpha2[-1], CIII2[-1], CI2[-1],sampled_params, dt, t)
        A_MII2.append(A_MII_next)
        I2.append(I_next)
        beta2.append(beta_next)
        A_MC2.append(A_MC_next)
        A_F2.append(A_F_next)
        A_M2.append(A_M_next)
        A_Malpha2.append(A_Malpha_next)
        CIII2.append(CIII_next)
        CI2.append(CI_next)
        if scenario == 1:
            outputs.append((A_MII1[-1], I1[-1], beta1[-1], A_MC1[-1], A_F1[-1], A_M1[-1], A_Malpha1[-1], CIII1[-1], CI1[-1]))
        elif scenario == 2:
            outputs.append((A_MII2[-1], I2[-1], beta2[-1], A_MC2[-1], A_F2[-1], A_M2[-1], A_Malpha2[-1], CIII2[-1], CI2[-1]))
    return np.array(outputs)
## Rest of your code for sampling, running the model, and performing sensitivity analysis goes here.

# Scenario 1
try:
    with open('model_outputs_scenario1.json', 'r') as f:
        outputs1 = np.array(json.load(f))
except FileNotFoundError:
    outputs1 = np.array([model_output(params, scenario=1) for params in param_values])
    with open('model_outputs_scenario1.json', 'w') as f:
        json.dump(outputs1.tolist(), f)

# Scenario 2
try:
    with open('model_outputs_scenario2.json', 'r') as f:
        outputs2 = np.array(json.load(f))
except FileNotFoundError:
    outputs2 = np.array([model_output(params, scenario=2) for params in param_values])
    with open('model_outputs_scenario2.json', 'w') as f:
        json.dump(outputs2.tolist(), f)


############ SCENARIO 1 ANALYSIS. ###########
# Scenario 1
with open('model_outputs_scenario1.json', 'r') as f:
    outputs1 = np.array(json.load(f))

print("shape of outputs1:",np.shape(outputs1))
# Reshape outputs1 to have dimensions (number of samples, number of outputs)
outputs1_reshaped = outputs1.reshape(outputs1.shape[0], -1)


# Perform Sobol sensitivity analysis for scenario 1
Sobol_indices_scenario1 = sobol.analyze(problem, outputs1_reshaped, calc_second_order=True, print_to_console=False)

# Save Sobol indices for Scenario 1 to a JSON file
with open('sobol_indices_scenario1.json', 'w') as json_file:
    json.dump(Sobol_indices_scenario1, json_file)

# Print the Sobol indices (first-order and total-order indices) for scenario 1
print("Scenario 1 Sobol Indices (First Order):", Sobol_indices_scenario1['S1'])
print("Scenario 1 Sobol Indices (Total Order):", Sobol_indices_scenario1['ST'])


############ SCENARIO 2 ANALYSIS. ###########
# # Scenario 2
# with open('model_outputs_scenario2.json', 'r') as f:
#     outputs2 = np.array(json.load(f))


# # Perform Sobol sensitivity analysis for scenario 2
# Sobol_indices_scenario2 = sobol.analyze(problem, outputs2, calc_second_order=True, print_to_console=False)

# # Save Sobol indices for Scenario 2 to a JSON file
# with open('sobol_indices_scenario2.json', 'w') as json_file:
#     json.dump(Sobol_indices_scenario2, json_file)

# # Print the Sobol indices (first-order and total-order indices) for scenario 2
# print("Scenario 2 Sobol Indices (First Order):", Sobol_indices_scenario2['S1'])
# print("Scenario 2 Sobol Indices (Total Order):", Sobol_indices_scenario2['ST'])

