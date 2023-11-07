import numpy as np
from SALib.sample import saltelli
import json
from param_ranges import boundfunc
import os 


# Create a directory to store pickle files
output_folder = "SA_results"
os.makedirs(output_folder, exist_ok=True)


def A_MII1_func(mu1, A_MII0, t):
    A_MII1 = np.exp(-mu1 * t) * A_MII0
    return A_MII1


def A_MII2_func(mu1, A_MII0, omega1, t):
    A_MII2 = A_MII0 * np.exp(-mu1 * t) * np.cos(omega1 * t)
    return A_MII2


def A_Malpha_func(rho2, A_M, A_Malpha0, t):
    A_Malpha = rho2 * A_M * t * A_Malpha0
    return A_Malpha


# Scenario 1 equations
def scenario1_equations(A_MII, I, beta, A_MC, A_F, A_M, A_Malpha, CIII, CI, parameters, dt, t):
    A_MII_next = A_MII1_func(parameters['mu1'], parameters['A_MII0'], t)
    I_next = I + dt * (
                -parameters['k1'] * parameters['upsilon1'] * np.exp(-parameters['upsilon1'] * t) + parameters['k5'] *
                parameters['gamma'] * A_MC - parameters['mu2'] * I)
    beta_next = beta + dt * (
                parameters['k2'] * parameters['upsilon2'] * np.exp(parameters['upsilon2'] * t) + parameters['k3'] *
                parameters['gamma'] * A_MII + parameters['k4'] * parameters['gamma'] * A_MC - parameters['mu3'] * beta)
    A_MC_next = A_MC + dt * (A_MC * I * parameters['lambda3'] * parameters['zeta'] - parameters['mu4'] * A_MC)
    A_F_next = A_F + dt * (parameters['lambda2'] * parameters['zeta'] * I * A_F + parameters['lambda1'] * parameters[
        'zeta'] * beta * A_F - parameters['rho1'] * A_F - parameters['mu5'] * A_F)
    A_M_next = A_M + dt * (
                parameters['rho1'] * A_F + parameters['lambda1'] * parameters['zeta'] * beta * A_F + parameters[
            'lambda4'] * parameters['zeta'] * A_F * A_M - parameters['mu6'] * A_M)
    A_Malpha_next = A_Malpha_func(parameters['rho2'], A_M, parameters['A_Malpha0'], t)
    CIII_next = CIII + dt * (
                parameters['k6'] * parameters['gamma'] * A_F + parameters['k9'] * parameters['gamma'] * A_M -
                parameters['rho3'] * CIII - parameters['mu7'] * CIII)
    CI_next = CI + dt * (parameters['rho3'] * CIII + parameters['k8'] * parameters['gamma'] * A_M + parameters['k3'] *
                         parameters['gamma'] * A_Malpha + parameters['k7'] * parameters['gamma'] * A_F * parameters[
                             'k10'] * parameters['gamma'] * A_Malpha - parameters['mu8'] * CI)
    return A_MII_next, I_next, beta_next, A_MC_next, A_F_next, A_M_next, A_Malpha_next, CIII_next, CI_next


# Scenario 2 equations
def scenario2_equations(A_MII, I, beta, A_MC, A_F, A_M, A_Malpha, CIII, CI, parameters, dt, t):
    A_MII_next = A_MII2_func(parameters['mu1'], parameters['A_MII0'], parameters['omega1'], t)
    I_next = I + dt * (-parameters['k1'] * parameters['upsilon3'] * np.exp(-parameters['upsilon3'] * t) * np.cos(
        parameters['omega2'] * t)
                       - parameters['k1'] * np.exp(-parameters['upsilon3'] * t) * parameters['omega2'] * np.sin(
                parameters['omega2'] * t) + parameters['k5'] * parameters['gamma'] * A_MC - parameters['mu2'] * I)
    beta_next = beta + dt * (parameters['k2'] * np.exp(-parameters['upsilon4'] * t) * parameters['omega3'] * np.cos(
        parameters['omega3'] * t)
                             - parameters['k2'] * parameters['upsilon4'] * np.exp(-parameters['upsilon4'] * t) * np.sin(
                parameters['omega3'] * t)
                             + parameters['k3'] * parameters['gamma'] * A_MII + parameters['k4'] * parameters[
                                 'gamma'] * A_MC - parameters['mu3'] * beta)
    A_MC_next = A_MC + dt * (A_MC * I * parameters['lambda3'] * parameters['zeta'] - parameters['mu4'] * A_MC)
    A_F_next = A_F + dt * (parameters['lambda2'] * parameters['zeta'] * I * A_F - parameters['lambda1'] * parameters[
        'zeta'] * beta * A_F - parameters['rho1'] * A_F - parameters['mu5'] * A_F)
    A_M_next = A_M + dt * (
                parameters['rho1'] * A_F + parameters['lambda1'] * parameters['zeta'] * beta * A_F + parameters[
            'lambda4'] * parameters['zeta'] * A_F * A_M - parameters['mu6'] * A_M)
    A_Malpha_next = A_Malpha_func(parameters['rho2'], A_M, parameters['A_Malpha0'], t)
    CIII_next = CIII + dt * (
                parameters['k6'] * parameters['gamma'] * A_F + parameters['k9'] * parameters['gamma'] * A_M -
                parameters['rho3'] * CIII - parameters['mu7'] * CIII)
    CI_next = CI + dt * (parameters['rho3'] * CIII + parameters['k8'] * parameters['gamma'] * A_M + parameters['k3'] *
                         parameters['gamma'] * A_Malpha + parameters['k7'] * parameters['gamma'] * A_F * parameters[
                             'k10'] * parameters['gamma'] * A_Malpha - parameters['mu8'] * CI)
    return A_MII_next, I_next, beta_next, A_MC_next, A_F_next, A_M_next, A_Malpha_next, CIII_next, CI_next


dt = 1/(60*24)
dt_hour = 1/60
# Production Parameters 
k1 = 2.34 * 10**(-5) * dt 
k2 = 1 * 10**(-5) * dt 
k3 = 8.80 * 10**(-5) * dt 
k4 = 0.00001 * dt 
k5 = 0.0005 * dt 
k6 = 30 * dt 
k7 = 1 * dt 
k8 = 50 * dt 
k9 = 30 * dt 
k10 = 20 * dt 

# Conversion parameters
gamma = 10**(-5) 
zeta = 10**(5) 

# Increase in proliferation parameters
lambda1 = 100 * dt 
lambda2 = 400 * dt 
lambda3 = 300 * dt 
lambda4 = 10**(-7) * dt 

# Transition parameters
rho1 = 5 * dt
rho2 = 3 * dt
rho3 = 18 * dt 

# # Decay parameters
mu1 = 2 * dt 
mu2 = 12 * dt 
mu3 = 10 * dt 
mu4 = 11 * dt 
mu5 = 10 * dt 
mu6 = 10 * dt 
mu7 = 5 * dt 
mu8 = 1 * dt 

# sinusoidal parameters
upsilon1 = -0.001 # negative value
upsilon2 = -0.1 # positive or negative
upsilon3 = 0.001 
upsilon4 = 0.001
omega1 = 1*np.pi *dt 
omega2 = 100*np.pi *dt 
omega3 = 80*np.pi *dt 

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
    'k1': k1, 'k2': k2, 'k3': k3, 'k4': k4, 'k5': k5, 'k6': k6, 'k7': k7, 'k8': k8, 'k9': k9, 'k10': k10,
    'gamma': gamma, 'zeta': zeta,
    'lambda1': lambda1, 'lambda2': lambda2, 'lambda3': lambda3, 'lambda4': lambda4,
    'rho1': rho1, 'rho2': rho2, 'rho3': rho3,
    'mu1': mu1, 'mu2': mu2, 'mu3': mu3, 'mu4': mu4, 'mu5': mu5, 'mu6': mu6, 'mu7': mu7, 'mu8': mu8,
    'upsilon1': upsilon1, 'upsilon2': upsilon2, 'upsilon3': upsilon3, 'upsilon4': upsilon4,
    'omega1': omega1, 'omega2': omega2, 'omega3': omega3,
    'A_MII0': A_MII0, 'I0': I0, 'beta0': beta0, 'A_MC0': A_MC0, 'A_F0': A_F0, 'A_M0': A_M0, 'A_Malpha0': A_Malpha0,
    'CIII0': CIII0, 'CI0': CI0
}

# Define bounds
bounds = boundfunc("array")

# Create the problem dictionary with bounds
problem = {
    'num_vars': len(initial_parameters),
    'names': list(initial_parameters.keys()),
    'bounds': bounds
}

# Sample params within bounds
param_values = saltelli.sample(problem, 128, calc_second_order=True)
param_values_list = param_values.tolist()
# Save
with open('sampled_params.json', 'w') as f:
    json.dump(param_values_list, f)


# Define a function that runs your model with given parameters and returns the output of interest
def model_output(params, cou, totalcou, parameters):
    outputs1 = []
    outputs2 = []
    outputs125 = []
    outputs225 = []
    outputs150 = []
    outputs250 = []
    outputs175 = []
    outputs275 = []
    # print((cou + 1) / totalcou)
    for params in param_values:
        # Convert params to a dictionary with parameter names as keys
        sampled_params = {param_name: param_value for param_name, param_value in zip(initial_parameters.keys(), params)}
    # print(sampled_params.values(), parameters.values())
    # Time parameters
    weeks = 1
    n_days_in_week = 7
    t_max = weeks * n_days_in_week  # Maximum simulation time(weeks)
    dt = 1 / (60 * 24)

    # Forward Euler method
    timesteps = int(t_max / dt)
    time = np.linspace(0, t_max, timesteps)
    # print(time, len(time))
    # print(IM)

    # Initialize arrays for results
    A_MII1 = sampled_params['A_MII0']
    I1 = sampled_params['I0']
    beta1 = sampled_params['beta0']
    A_MC1 = sampled_params['A_MC0']
    A_F1 = sampled_params['A_F0']
    A_M1 = sampled_params['A_M0']
    A_Malpha1 = sampled_params['A_Malpha0']
    CIII1 = sampled_params['CIII0']
    CI1 = sampled_params['CI0']

    A_MII2 = sampled_params['A_MII0']
    I2 = sampled_params['I0']
    beta2 = sampled_params['beta0']
    A_MC2 = sampled_params['A_MC0']
    A_F2 = sampled_params['A_F0']
    A_M2 = sampled_params['A_M0']
    A_Malpha2 = sampled_params['A_Malpha0']
    CIII2 = sampled_params['CIII0']
    CI2 = sampled_params['CI0']

    startinc = 0
    # Perform simulation for both scenarios using forward Euler method
    for i in range(0, timesteps + 1):
        t = i * dt
        savetimes = [int(i*timvals) for timvals in [0.25,0.5,0.75,1]]
        # Scenario 1
        A_MII1, I1, beta1, A_MC1, A_F1, A_M1, A_Malpha1, CIII1, CI1 = scenario1_equations(A_MII1, I1, beta1, A_MC1, A_F1, A_M1, A_Malpha1, CIII1, CI1, sampled_params, dt, t)

        # Scenario 2
        A_MII2, I2, beta2, A_MC2, A_F2, A_M2, A_Malpha2, CIII2, CI2 = scenario2_equations(A_MII2, I2, beta2, A_MC2, A_F2, A_M2, A_Malpha2, CIII2, CI2, sampled_params, dt, t)

        if i in savetimes:
            if startinc == 0:
                outputs125.append((A_MII1, I1, beta1, A_MC1, A_F1, A_M1, A_Malpha1, CIII1, CI1))
                outputs225.append((A_MII2, I2, beta2, A_MC2, A_F2, A_M2, A_Malpha2, CIII2, CI2))
            if startinc == 1:
                outputs150.append((A_MII1, I1, beta1, A_MC1, A_F1, A_M1, A_Malpha1, CIII1, CI1))
                outputs250.append((A_MII2, I2, beta2, A_MC2, A_F2, A_M2, A_Malpha2, CIII2, CI2))
            if startinc == 2:
                outputs175.append((A_MII1, I1, beta1, A_MC1, A_F1, A_M1, A_Malpha1, CIII1, CI1))
                outputs275.append((A_MII2, I2, beta2, A_MC2, A_F2, A_M2, A_Malpha2, CIII2, CI2))
            if startinc == 3:
                outputs1.append((A_MII1, I1, beta1, A_MC1, A_F1, A_M1, A_Malpha1, CIII1, CI1))
                outputs2.append((A_MII2, I2, beta2, A_MC2, A_F2, A_M2, A_Malpha2, CIII2, CI2))
            startinc+=1
    return np.array(outputs125), np.array(outputs225), np.array(outputs150), np.array(outputs250), np.array(outputs175), np.array(outputs275), np.array(outputs1), np.array(outputs2)



# Load results from the specified experiment
with open(os.path.join(output_folder, "sampled_params.pkl"), "rb") as f:
    param_values = pickle.load(f)
print("Samples loaded!")

# Both scenarios
runningnow = 0
for cou, params in enumerate(param_values):
    output = model_output(params, cou, len(param_values), initial_parameters)
    with open(os.path.join(output_folder, 'model_outputs_iter{}.pkl'.format(cou)), 'wb') as f:
        pickle.dump(list(output), f)
    runningnow += (1/len(param_values))
    percentage = runningnow * 100
    print(f'{percentage} of data saved!'.format())

def compile(directory_path):
    # List to store the loaded data from .pkl files
    compiled_data = []

    # Iterate through files in the directory
    for filename in os.listdir(directory_path):
        if filename.startswith('model_outputs_iter') and filename.endswith('.pkl'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'rb') as file:
                # Load data from the current file
                data = pickle.load(file)
                # Append the loaded data to the compiled_data list
                compiled_data.extend(data)

    # Save the compiled data as a new .pkl file
    output_file_path = 'compiled_model_outputs.pkl'
    with open(output_file_path, 'wb') as output_file:
        # Dump the compiled_data list into the output .pkl file
        pickle.dump(compiled_data, output_file)

# compile(output_folder)
