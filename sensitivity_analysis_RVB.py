import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
import json
import matplotlib.pyplot as plt
import os 
from param_ranges import boundfunc

# Specify the folder path where you want to save the files
output_folder_results_Y = "results_Y"


# Specify the folder path where you want to save the files
output_folder_results_D = "results_D"

# Check if the folder exists, if not, create it
if not os.path.exists(output_folder_results_Y):
    os.makedirs(output_folder_results_Y)

# Check if the folder exists, if not, create it
if not os.path.exists(output_folder_results_D):
    os.makedirs(output_folder_results_D)



def A_MII1_func(mu1, A_MII0, t):
    A_MII1 = np.exp(-mu1 * t) * A_MII0
    return A_MII1 

# a = A_MII1_func(k1, mu1, A_MII0, time)

def A_MII2_func(mu1, A_MII0, omega1, t):

    A_MII2 = A_MII0*np.exp(-mu1 * t) * np.cos(omega1 * t)
    return A_MII2 

def A_Malpha_func(rho2, A_M, A_Malpha0, t):
    A_Malpha = rho2 * A_M * t * A_Malpha0
    return A_Malpha



# Scenario 1 equations
def scenario1_equations(A_MII, I, beta, A_MC, A_F, A_M, A_Malpha, CIII, CI, parameters, dt, t):
    A_MII_next = A_MII1_func(parameters['mu1'], parameters['A_MII0'], t)
    I_next = I + dt  * (-parameters['k1'] * parameters['upsilon1'] * np.exp(-parameters['upsilon1'] * t) + parameters['k5'] * parameters['gamma'] * A_MC - parameters['mu2'] * I)
    beta_next = beta + dt * (parameters['k2'] * parameters['upsilon2'] * np.exp(parameters['upsilon2'] * t) + parameters['k3'] * parameters['gamma'] * A_MII + parameters['k4'] * parameters['gamma'] * A_MC - parameters['mu3'] * beta)
    A_MC_next = A_MC + dt * (A_MC * I * parameters['lambda3'] * parameters['zeta'] - parameters['mu4'] * A_MC)
    A_F_next = A_F + dt * (parameters['lambda2'] * parameters['zeta'] * I * A_F + parameters['lambda1'] * parameters['zeta'] * beta * A_F - parameters['rho1'] * A_F - parameters['mu5'] * A_F)
    A_M_next = A_M + dt * (parameters['rho1'] * A_F + parameters['lambda1'] * parameters['zeta'] * beta * A_F + parameters['lambda4'] * parameters['zeta'] * A_F * A_M - parameters['mu6'] * A_M)
    A_Malpha_next = A_Malpha_func(parameters['rho2'], A_M, parameters['A_Malpha0'], t)
    CIII_next = CIII + dt * (parameters['k6'] * parameters['gamma'] * A_F + parameters['k9'] * parameters['gamma'] * A_M - parameters['rho3'] * CIII - parameters['mu7'] * CIII)
    CI_next = CI + dt * (parameters['rho3'] * CIII + parameters['k8'] * parameters['gamma'] * A_M + parameters['k3'] * parameters['gamma'] * A_Malpha + parameters['k7'] * parameters['gamma'] * A_F * parameters['k10'] * parameters['gamma'] * A_Malpha - parameters['mu8'] * CI)
    return A_MII_next, I_next, beta_next, A_MC_next, A_F_next, A_M_next, A_Malpha_next, CIII_next, CI_next

# Scenario 2 equations
def scenario2_equations(A_MII, I, beta, A_MC, A_F, A_M, A_Malpha, CIII, CI, parameters, dt, t):
    A_MII_next = A_MII2_func(parameters['mu1'], parameters['A_MII0'], parameters['omega1'], t)
    I_next = I + dt  * (-parameters['k1'] * parameters['upsilon3'] * np.exp(-parameters['upsilon3'] * t) * np.cos(parameters['omega2'] * t)
                       - parameters['k1'] * np.exp(-parameters['upsilon3'] * t) * parameters['omega2'] * np.sin(parameters['omega2'] * t) + parameters['k5'] * parameters['gamma'] * A_MC- parameters['mu2'] * I)
    beta_next = beta + dt  * (parameters['k2'] * np.exp(-parameters['upsilon4'] * t) * parameters['omega3'] * np.cos(parameters['omega3'] * t)
                             - parameters['k2'] * parameters['upsilon4'] * np.exp(-parameters['upsilon4'] * t) * np.sin(parameters['omega3'] * t)
                             + parameters['k3'] * parameters['gamma'] * A_MII + parameters['k4'] * parameters['gamma'] * A_MC - parameters['mu3'] * beta)
    A_MC_next = A_MC + dt * (A_MC * I * parameters['lambda3'] * parameters['zeta'] - parameters['mu4'] * A_MC)
    A_F_next = A_F + dt * (parameters['lambda2'] * parameters['zeta'] * I * A_F - parameters['lambda1'] * parameters['zeta'] * beta * A_F - parameters['rho1'] * A_F - parameters['mu5'] * A_F)
    A_M_next = A_M + dt * (parameters['rho1'] * A_F + parameters['lambda1'] * parameters['zeta'] * beta * A_F + parameters['lambda4'] * parameters['zeta'] * A_F * A_M - parameters['mu6'] * A_M)
    A_Malpha_next = A_Malpha_func(parameters['rho2'], A_M, parameters['A_Malpha0'], t)
    CIII_next = CIII + dt * (parameters['k6'] * parameters['gamma'] * A_F + parameters['k9'] * parameters['gamma'] * A_M - parameters['rho3'] * CIII - parameters['mu7'] * CIII)
    CI_next = CI + dt * (parameters['rho3'] * CIII + parameters['k8'] * parameters['gamma'] * A_M + parameters['k3'] * parameters['gamma'] * A_Malpha + parameters['k7'] * parameters['gamma'] * A_F * parameters['k10'] * parameters['gamma'] * A_Malpha - parameters['mu8'] * CI)
    return A_MII_next, I_next, beta_next, A_MC_next, A_F_next, A_M_next, A_Malpha_next, CIII_next, CI_next

# Define a function that runs your model with given parameters and returns the output of interest
def model_output(params, var):
    # Unpack initial values from the params dictionary
    A_MII0, I0, beta0, A_MC0, A_F0, A_M0, A_Malpha0, CIII0, CI0 = \
        params['A_MII0'], params['I0'], params['beta0'], params['A_MC0'], \
        params['A_F0'], params['A_M0'], params['A_Malpha0'], params['CIII0'], \
        params['CI0']

    # Time parameters
    weeks = 30
    n_days_in_week = 7
    t_max =  weeks * n_days_in_week # Maximum simulation time(weeks)
    dt = 1/(60*24)

    # Forward Euler method
    timesteps = int(t_max/dt)
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

    # Perform simulation for both scenarios using forward Euler method
    for i in range(1, timesteps + 1):
        t = i * dt
        
        # Scenario 1
        A_MII_next, I_next, beta_next, A_MC_next, A_F_next, A_M_next, A_Malpha_next, CIII_next, CI_next = \
            scenario1_equations(A_MII1[-1], I1[-1], beta1[-1], A_MC1[-1], A_F1[-1], A_M1[-1], A_Malpha1[-1], CIII1[-1], CI1[-1], params, dt, t)
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
            scenario2_equations(A_MII2[-1], I2[-1], beta2[-1], A_MC2[-1], A_F2[-1], A_M2[-1], A_Malpha2[-1], CIII2[-1], CI2[-1], params, dt, t)
        A_MII2.append(A_MII_next)
        I2.append(I_next)
        beta2.append(beta_next)
        A_MC2.append(A_MC_next)
        A_F2.append(A_F_next)
        A_M2.append(A_M_next)
        A_Malpha2.append(A_Malpha_next)
        CIII2.append(CIII_next)
        CI2.append(CI_next)
    
    # Create a dictionary for easier variable lookup
    variables = {
        'A_MII1': A_MII1[1:],
        'A_MII2': A_MII2[1:],
        'I1': I1[1:],
        'I2': I2[1:],
        'beta1': beta1[1:],
        'beta2': beta2[1:],
        'A_MC1': A_MC1[1:],
        'A_MC2': A_MC2[1:],
        'A_F1': A_F1[1:],
        'A_F2': A_F2[1:],
        'A_M1': A_M1[1:],
        'A_M2': A_M2[1:],
        'A_Malpha1': A_Malpha1[1:],
        'A_Malpha2': A_Malpha2[1:],
        'CIII1': CIII1[1:],
        'CIII2': CIII2[1:],
        'CI1': CI1[1:],
        'CI2': CI2[1:]
    }
    
    # Return the requested variable
    return variables.get(var, None)


def generate_samples(problem):
    # Check if saved samples exist, if not, generate new samples and save them
    try:
        with open('sampled_params.json', 'r') as f:
            param_values_list = json.load(f)
        param_values = np.array(param_values_list)
    except FileNotFoundError:
        param_values = saltelli.sample(problem, 128, calc_second_order=True)
        param_values_list = param_values.tolist()  # Convert NumPy array to list for JSON serialization
        with open('sampled_params.json', 'w') as f:
            json.dump(param_values_list, f)
    print("Samples generated !")
    return 


def run_Y(output_folder, problem, var, parameters):
    sampled_params_file = 'sampled_params.json'
    try:
        with open(sampled_params_file, 'r') as f:
            sampled_params_list = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Sampled parameters file not found.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON in sampled parameters file: {e}")

    # Ensure the number of sampled parameters matches the expected count
    if len(sampled_params_list) != 11264:
        raise ValueError("Invalid number of sampled parameters.")
    
    Yresult = []
    for sampled_param_values in sampled_params_list:
        # Ensure the current sampled parameters have the expected length (43 parameters)
        if len(sampled_param_values) != 43:
            raise ValueError("Invalid number of parameters in the sampled set.")
        
        # Update the parameters dictionary with the current set of parameters
        for key, value in zip(parameters.keys(), sampled_param_values):
            parameters[key] = value
    
        y_result = model_output(parameters, var)
        
        Yresult.append(y_result)
    
    # Convert to NumPy array and save to file
    Yresult = np.array(Yresult)
    Yresult_file_path = os.path.join(output_folder, '{}_Yresults.json'.format(var))
    with open(Yresult_file_path, 'w') as f:
        json.dump(Yresult.tolist(), f)

    return Yresult


def run_D(output_folder, problem, var):
    Yresult = run_Y(output_folder, problem, var)  # Get Yresult from run_Y function

    output_file_path = os.path.join(output_folder, '{}_Dindices.json'.format(var))

    try:
        with open(output_file_path, 'r') as f:
            output = json.load(f)
        # Validate output data, ensuring it's not empty
        if not output:
            raise ValueError("Empty or invalid output data loaded from file.")
    except (FileNotFoundError, ValueError):
        # Perform Sobol analysis if output file not found or contains invalid data
        total_indices, first_order_indices, second_order_indices = sobol.analyze(problem, Yresult, calc_second_order=True)
        output = {
            "total_indices": total_indices.tolist(),
            "first_order_indices": first_order_indices.tolist(),
            "second_order_indices": second_order_indices.tolist()
        }
        with open(output_file_path, 'w') as f:
            json.dump(output, f)

    return output





# def run_D(output_folder, problem, Yresult, var):
#     output_file_path = os.path.join(output_folder, '{}_Dindices.json'.format(var))
#     try:
#         with open(output_file_path, 'r') as f:
#             output = np.array(json.load(f))
#     except FileNotFoundError:
#         output = sobol.analyze(problem, Yresult, calc_second_order=True, print_to_console=False)
#         with open(output_file_path, 'w') as f:
#             json.dump(list(output), f)

#     return


dt = 1/(60*24)
# Production Parameters 
# k1 = 2.34 * 10**(-5) * dt #2.34 * 10**(-5) * dt *rho2 https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009839
k1 = 2.34 * 10**(-5) * dt #2.34 * 10**(-5) * dt  combi model ****
k2 = 1 * 10**(-5) * dt # 1 * 10**(-5) * dt 
k3 = 8.80 * 10**(-5) * dt # 28.0 * 10**(-5) * dt day combi model *****
k4 = 0.00001 * dt # 0.00001 * dt
k5 = 0.0005 * dt # 0.0005 * dt
k6 = 30 * dt #30 * dt  k1 and k2 between 30-60 https://zero.sci-hub.se/3771/b68709ea5f5640da4199e36ff25ef036/cumming2009.pdf
k7 = 1 * dt # 1 * dt https://link.springer.com/article/10.1007/s11538-012-9751-z/tables/1
k8 = 50 * dt # 50 * dt
k9 = 30 * dt # 30 * dt
k10 = 20 * dt # 2000000 * dt production of CI forced by myofibroblasts https://www.sciencedirect.com/science/article/pii/S0045782516302857?casa_token=ByHEzHgojSEAAAAA:XNdfPARqEPtiO3rcqb0jo9d--utWdu-swPxNKOLyK5huphzY4TcRxiVo4c4yzCASMY-tOswVpzY#br000425

# Conversion parameters
gamma = 10**(-5) 
zeta = 10**(5) # fixed

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




parameters = {
      'k1': k1, 'k2': k2, 'k3': k3, 'k4': k4, 'k5': k5, 'k6': k6, 'k7': k7, 'k8': k8, 'k9': k9, 'k10': k10, 
      'gamma': gamma,'zeta': zeta,
      'lambda1': lambda1, 'lambda2': lambda2, 'lambda3': lambda3, 'lambda4': lambda4,
      'rho1': rho1, 'rho2': rho2, 'rho3': rho3,
      'mu1': mu1, 'mu2': mu2, 'mu3': mu3, 'mu4': mu4, 'mu5': mu5, 'mu6': mu6, 'mu7': mu7, 'mu8': mu8,
      'upsilon1': upsilon1, 'upsilon2': upsilon2, 'upsilon3': upsilon3, 'upsilon4': upsilon4,
      'omega1': omega1, 'omega2': omega2, 'omega3': omega3,
      'A_MII0': A_MII0, 'I0': I0, 'beta0': beta0, 'A_MC0': A_MC0, 'A_F0': A_F0, 'A_M0': A_M0, 'A_Malpha0': A_Malpha0,
      'CIII0': CIII0, 'CI0': CI0
  }


def check_invalid_bounds(bounds_dict):
    invalid_params = []
    for param, (min_value, max_value) in bounds_dict.items():
        if min_value > max_value:
            invalid_params.append(param)
    return invalid_params


bounds = boundfunc("array")

# invalid_params = check_invalid_bounds(bounds)
# print("Parameters with invalid bounds:", invalid_params)



# # Convert parameter ranges to SALib bounds format
# bounds = []
# for param, values in parameter_ranges.items():
#     bounds.append(values)
# print(np.asarray(bounds))
# print(bounds)
# Create the problem dictionary with bounds based on the initial guess and increment


problem = {
        'num_vars': len(parameters),
        'names': list(parameters.keys()),
        'bounds': bounds
    }


# run_Y(output_folder = output_folder_results_Y, parameters = parameters, var = "A_MII1")
variable = "I1"
# run_D(output_folderY = output_folder_results_Y, output_folderD = output_folder_results_D, problem = problem, var = variable, parameters = parameters)

# def run(parameters, bounds, var):
#     problem = {
#         'num_vars': len(parameters),
#         'names': list(parameters.keys()),
#         'bounds': bounds
#     }
#     generate_samples(problem)

# run_Y(output_folder, problem, var, parameters)
run_Y(output_folder = output_folder_results_Y, problem = problem, var = variable, parameters = parameters)
# run_Y(output_folder, params, var)
#     run_D(output_folder, problem, Yresult, var)
