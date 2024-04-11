import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.cm as cm
import pickle
import os



# Create a directory to store pickle files
output_folder2 = "pickle_plasma2"
os.makedirs(output_folder2, exist_ok=True)

# Create a directory to store pickle files
output_folder3 = "pickle_plasma3"
os.makedirs(output_folder3, exist_ok=True)

# Create a directory to store pickle files
output_folder4 = "best_init_params"
os.makedirs(output_folder4, exist_ok=True)

# Create a directory to store pickle files
output_folder5 = "pickle_plasma4"
os.makedirs(output_folder5, exist_ok=True)


def A_MII1_func(mu1, A_MII0, t):
    A_MII1 = np.exp(-mu1 * t) * A_MII0
    return A_MII1 

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

def initialize_problem_from_data(val, var):
    if val == "max":
        val = "best"
    if val == "min":
        val = "worst"

    with open(os.path.join(output_folder4, f"{val}_initvalprob_{var}.pkl"), "rb") as f:
        loaded_results = pickle.load(f)
    print(loaded_results)
    initial_values = loaded_results['initial_values'] 

    dt = 1/(60*24)
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

    # Activation parameters
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
    upsilon1 = -0.001 
    upsilon2 = -0.1
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

      # Time parameters
    weeks = 30
    n_days_in_week = 7
    t_max =  weeks * n_days_in_week # Maximum simulation time(weeks)

    # Forward Euler method
    timesteps = int(t_max/dt)
    time = np.linspace(0, t_max, timesteps)



    # Initialize arrays for results
    A_MII1 = [initial_values['A_MII0']]
    I1 = [initial_values['I0']]
    beta1 = [initial_values['beta0']]
    A_MC1 = [initial_values['A_MC0']]
    A_F1 = [initial_values['A_F0']]
    A_M1 = [initial_values['A_M0']]
    A_Malpha1 = [initial_values['A_Malpha0']]
    CIII1 = [initial_values['CIII0']]
    CI1= [initial_values['CIII0']]

    # Initialize arrays for results
    A_MII2 = [initial_values['A_MII0']]
    I2 = [initial_values['I0']]
    beta2 = [initial_values['beta0']]
    A_MC2 = [initial_values['A_MC0']]
    A_F2 = [initial_values['A_F0']]
    A_M2 = [initial_values['A_M0']]
    A_Malpha2 = [initial_values['A_Malpha0']]
    CIII2 = [initial_values['CIII0']]
    CI2 = [initial_values['CIII0']]

    # Perform simulation for both scenarios using forward Euler method
    for i in range(1, timesteps + 1):
        t = i * dt
        
        # Scenario 1
        A_MII_next, I_next, beta_next, A_MC_next, A_F_next, A_M_next, A_Malpha_next, CIII_next, CI_next = \
            scenario1_equations(A_MII1[-1], I1[-1], beta1[-1], A_MC1[-1], A_F1[-1], A_M1[-1], A_Malpha1[-1], CIII1[-1], CI1[-1],parameters, dt, t)
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
            scenario2_equations(A_MII2[-1], I2[-1], beta2[-1], A_MC2[-1], A_F2[-1], A_M2[-1], A_Malpha2[-1], CIII2[-1], CI2[-1],parameters, dt, t)
        A_MII2.append(A_MII_next)
        I2.append(I_next)
        beta2.append(beta_next)
        A_MC2.append(A_MC_next)
        A_F2.append(A_F_next)
        A_M2.append(A_M_next)
        A_Malpha2.append(A_Malpha_next)
        CIII2.append(CIII_next)
        CI2.append(CI_next)

    # Return which variable is needed
    if var == "A_MII1":
        return A_MII1
    if var == "I1":
        return I1
    if var == "beta1":
        return beta1
    if var == "A_MC1":
        return A_MC1
    if var == "A_F1":
        return A_F1
    if var == "A_M1":
        return A_M1
    if var == "A_Malpha1":
        return A_Malpha1
    if var == "CIII1":
        return CIII1
    if var == "CI1":
        return CI1
    if var == "A_MII2":
        return A_MII2
    if var == "I2":
        return I2
    if var == "beta2":
        return beta2
    if var == "A_MC2":
        return A_MC2
    if var == "A_F2":
        return A_F2
    if var == "A_M2":
        return A_M2
    if var == "A_Malpha2":
        return A_Malpha2
    if var == "CIII2":
        return CIII2
    if var == "CI2":
        return CI2

def get_all_best_within_range(results_folder):
    keys = ["CIII1"]#,"CIII2","CI1","CI2","A_F1","A_M1","A_F2","A_M2", \
    # "A_Malpha1","A_Malpha2","A_MC1","A_MC2","A_MII1","A_MII2",\
    # "I1","I2","beta1","beta2"]
    # specify bounds for search
    bounds = [[0,1], [0, 1], [0, 2], [0, 2], [0, 5000], [0, 3000],
    [0, 5000], [0, 3000], [0, 1500], [0, 1500], [0, 5000], [0, 5000],
    [0, 5000], [0, 5000],[0, 10**(-5)], [0, 10**(-5)], [0, 10**(-5)], 
    [0, 10**(-5)]]
    for l in range(len(keys)):
        print("doing key {}".format(keys[l]))
        variable_of_interest = keys[l]  
        lower_bound = bounds[l][0]  # Specify the lower bound of the variable
        upper_bound = bounds[l][1]  # Specify the upper bound of the variable
        analyze_results(results_folder, variable_of_interest, lower_bound, upper_bound)

def perform_exp(nr_exp, output_folder):

    dt = 1/(60*24)
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


    # Time parameters
    weeks = 30
    n_days_in_week = 7
    t_max =  weeks * n_days_in_week # Maximum simulation time(weeks)

    # Forward Euler method
    timesteps = int(t_max/dt)
    time = np.linspace(0, t_max, timesteps)



    # Define the parameter ranges for randomization
    parameter_ranges = {
        'A_MII0': (0, 5000), # (500, 3500)
        'I0': (1e-11, 5e-5), # (1e-9, 5e-7)
        'beta0': (1e-11, 5e-5), # (1e-9, 5e-7)
        'A_MC0': (0, 5000), # (500, 3500)
        'A_F0': (0,10000), # (0,2800)
        'A_M0': (0,10000), # (0,1000)
        'A_Malpha0': (0,5000), # (0,500)
        'CIII0': (0,1), # (0,1)
        'CI0': (0,1) # (0,0.1)
    }
    # Initialize the number of experiments (randomized initial values)
    num_experiments = nr_exp

    # Perform simulations for each experiment using forward Euler method
    for experiment in range(num_experiments):
        random_initial_values = {param: np.random.uniform(low, high) for param, (low, high) in parameter_ranges.items()}

        # Initialize arrays for results
        A_MII1 = [random_initial_values['A_MII0']]
        I1 = [random_initial_values['I0']]
        beta1 = [random_initial_values['beta0']]
        A_MC1 = [random_initial_values['A_MC0']]
        A_F1 = [random_initial_values['A_F0']]
        A_M1 = [random_initial_values['A_M0']]
        A_Malpha1 = [random_initial_values['A_Malpha0']]
        CIII1 = [random_initial_values['CIII0']]
        CI1= [random_initial_values['CIII0']]

        # Initialize arrays for results
        A_MII2 = [random_initial_values['A_MII0']]
        I2 = [random_initial_values['I0']]
        beta2 = [random_initial_values['beta0']]
        A_MC2 = [random_initial_values['A_MC0']]
        A_F2 = [random_initial_values['A_F0']]
        A_M2 = [random_initial_values['A_M0']]
        A_Malpha2 = [random_initial_values['A_Malpha0']]
        CIII2 = [random_initial_values['CIII0']]
        CI2 = [random_initial_values['CIII0']]

        # Perform simulation for both scenarios using forward Euler method
        for i in range(1, timesteps + 1):
            t = i * dt
            
            # Scenario 1
            A_MII_next, I_next, beta_next, A_MC_next, A_F_next, A_M_next, A_Malpha_next, CIII_next, CI_next = \
                scenario1_equations(A_MII1[-1], I1[-1], beta1[-1], A_MC1[-1], A_F1[-1], A_M1[-1], A_Malpha1[-1], CIII1[-1], CI1[-1],parameters, dt, t)
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
                scenario2_equations(A_MII2[-1], I2[-1], beta2[-1], A_MC2[-1], A_F2[-1], A_M2[-1], A_Malpha2[-1], CIII2[-1], CI2[-1],parameters, dt, t)
            A_MII2.append(A_MII_next)
            I2.append(I_next)
            beta2.append(beta_next)
            A_MC2.append(A_MC_next)
            A_F2.append(A_F_next)
            A_M2.append(A_M_next)
            A_Malpha2.append(A_Malpha_next)
            CIII2.append(CIII_next)
            CI2.append(CI_next)

        # Save results for each experiment in a pickle file
        results = {
        "time": time,
        "CIII1": CIII1,
        "CIII2": CIII2,
        "CI1": CI1,
        "CI2": CI2,
        "A_F1":A_F1,
        "A_M1":A_M1,
        "A_F2":A_F2,
        "A_M2":A_M2, 
        "A_Malpha1":A_Malpha1,
        "A_Malpha2":A_Malpha2,
        "A_MC1":A_MC1,
        "A_MC2":A_MC2,
        "A_MII1":A_MII1,
        "A_MII2":A_MII2,
        "I1":I1,
        "I2":I2,
        "beta1":beta1,
        "beta2":beta2,
        "initial_values1": {
            "A_MII0": A_MII1[0],
            "I0": I1[0],
            "beta0": beta1[0],
            "A_MC0": A_MC1[0],
            "A_F0": A_F1[0],
            "A_M0": A_M1[0],
            "A_Malpha0": A_Malpha1[0],
            "CIII0": CIII1[0],
            "CI0": CI1[0]
            },
        "initial_values2": {
            "A_MII0": A_MII2[0],
            "I0": I2[0],
            "beta0": beta2[0],
            "A_MC0": A_MC2[0],
            "A_F0": A_F2[0],
            "A_M0": A_M2[0],
            "A_Malpha0": A_Malpha2[0],
            "CIII0": CIII2[0],
            "CI0": CI2[0]
            }
        }
        
        with open(os.path.join(output_folder, f"experiment_{experiment}_results.pkl"), "wb") as f:
            pickle.dump(results, f)



#PLOT

def plot_exp1(nr_exp, out):
    # Load results from all experiments and plot
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    plasma_colormap = matplotlib.cm.get_cmap("nipy_spectral")
    colors = [plasma_colormap(x) for x in np.linspace(0.8, 0.15, num=nr_exp)]
    colors = [matplotlib.colors.to_hex(color) for color in colors]


    # Lists to store valid data for computing mean curve
    valid_CIII1_data = []
    valid_CIII2_data = []
    valid_CI1_data = []
    valid_CI2_data = []

    # Loop through all experiments
    for experiment_to_load, color in zip(range(nr_exp), colors):
        # Load results from the specified experiment
        with open(os.path.join(out, f"experiment_{experiment_to_load}_results.pkl"), "rb") as f:
            loaded_results = pickle.load(f)
        # print(loaded_results.keys())
        # continue
        # Check if CIII1 data is within y-limits
        if all(0<y_value <= 1 for y_value in loaded_results["CIII1"]):
            axs[0, 0].plot(loaded_results["time"], loaded_results["CIII1"][1:], color=color, alpha=0.2)
            valid_CIII1_data.append(loaded_results["CIII1"][1:])

        # Check if CIII2 data is within y-limits
        if all(0<y_value <= 1 for y_value in loaded_results["CIII2"]):
            axs[0, 1].plot(loaded_results["time"], loaded_results["CIII2"][1:], color=color, alpha=0.2)
            valid_CIII2_data.append(loaded_results["CIII2"][1:])

        # Check if CI1 data is within y-limits
        if all(0<y_value <= 2 for y_value in loaded_results["CI1"]):
            axs[1, 0].plot(loaded_results["time"], loaded_results["CI1"][1:], color=color, alpha=0.2)
            valid_CI1_data.append(loaded_results["CI1"][1:])

        # Check if CI2 data is within y-limits
        if all(0<y_value <= 2 for y_value in loaded_results["CI2"]):
            axs[1, 1].plot(loaded_results["time"], loaded_results["CI2"][1:], color=color, alpha=0.2)
            valid_CI2_data.append(loaded_results["CI2"][1:])
    
    # Compute mean curves
    mean_CIII1 = np.mean(valid_CIII1_data, axis=0)
    mean_CIII2 = np.mean(valid_CIII2_data, axis=0)
    mean_CI1 = np.mean(valid_CI1_data, axis=0)
    mean_CI2 = np.mean(valid_CI2_data, axis=0)

    max_CIII1 = initialize_problem_from_data("max", "CIII1")[1:]
    min_CIII1 = initialize_problem_from_data("min", "CIII1")[1:]
    max_CIII2 = initialize_problem_from_data("max", "CIII2")[1:]
    min_CIII2 = initialize_problem_from_data("min", "CIII2")[1:]
    max_CI1 = initialize_problem_from_data("max", "CI1")[1:]
    min_CI1 = initialize_problem_from_data("min", "CI1")[1:]
    max_CI2 = initialize_problem_from_data("max", "CI2")[1:]
    min_CI2 = initialize_problem_from_data("min", "CI2")[1:]

    # Plot mean curves with alpha=1 and increased linewidth
    axs[0, 0].plot(loaded_results["time"], mean_CIII1, label = 'Mean $C_{III1}$',color='black', alpha=1, linewidth=3)
    axs[0, 1].plot(loaded_results["time"], mean_CIII2, label = 'Mean $C_{III2}$',color='black', alpha=1, linewidth=3)
    axs[1, 0].plot(loaded_results["time"], mean_CI1, label = 'Mean $C_{I1}$', color='black', alpha=1, linewidth=3)
    axs[1, 1].plot(loaded_results["time"], mean_CI2, label = 'Mean $C_{I2}$', color='black', alpha=1, linewidth=3)

    # Plot maximum and minimum curves
    axs[0, 0].plot(loaded_results["time"], max_CIII1, label='Max $C_{III1}$', color='blue', alpha=1, linewidth=2)
    axs[0, 0].plot(loaded_results["time"], min_CIII1, label='Min $C_{III1}$', color='green', alpha=1, linewidth=2)
    axs[0, 1].plot(loaded_results["time"], max_CIII2, label='Max $C_{III2}$', color='blue', alpha=1, linewidth=2)
    axs[0, 1].plot(loaded_results["time"], min_CIII2, label='Min $C_{III2}$', color='green', alpha=1, linewidth=2)
    axs[1, 0].plot(loaded_results["time"], max_CI1, label='Max $C_{I1}$', color='blue', alpha=1, linewidth=2)
    axs[1, 0].plot(loaded_results["time"], min_CI1, label='Min $C_{I1}$', color='green', alpha=1, linewidth=2)
    axs[1, 1].plot(loaded_results["time"], max_CI2, label='Max $C_{I2}$', color='blue', alpha=1, linewidth=2)
    axs[1, 1].plot(loaded_results["time"], min_CI2, label='Min $C_{I2}$', color='green', alpha=1, linewidth=2)

    # Set titles and y-limits for subplots
    # axs[0, 0].set_title('CIII1')
    axs[0, 0].set_ylim(0, 1)
    axs[0, 0].legend(loc = 'upper center')
    axs[0, 0].set_ylabel('Collagen concentration')
    # axs[0, 1].set_title('CIII2')
    axs[0, 1].set_ylim(0, 1)
    axs[0, 1].legend(loc = 'upper center')
    axs[0, 1].yaxis.tick_right()
    # axs[1, 0].set_title('CI1')
    axs[1, 0].set_ylim(0, 2)
    axs[1, 0].legend(loc = 'upper center')
    axs[1, 0].set_xlabel('Time(Days)')
    axs[1, 0].set_ylabel('Collagen concentration')
    # axs[1, 1].set_title('CI2')
    axs[1, 1].set_ylim(0, 2)
    axs[1, 1].legend(loc = 'upper center')
    axs[1, 1].yaxis.tick_right()
    axs[1, 1].set_xlabel('Time(Days)')

    # Save the figure as PNG with specified DPI
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'output_figure_C_c1.png'), dpi=300)


def plot_exp2(nr_exp, out):
    # Load results from all experiments and plot
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    plasma_colormap = matplotlib.cm.get_cmap("bone")
    colors = [plasma_colormap(x) for x in np.linspace(0.8, 0.15, num=nr_exp)]
    colors = [matplotlib.colors.to_hex(color) for color in colors]

    # Lists to store valid data for computing mean curve
    valid_F1_data = []
    valid_F2_data = []
    valid_M1_data = []
    valid_M2_data = []

    # Loop through all experiments
    for experiment_to_load, color in zip(range(nr_exp), colors):
        # Load results from the specified experiment
        with open(os.path.join(out, f"experiment_{experiment_to_load}_results.pkl"), "rb") as f:
            loaded_results = pickle.load(f)
        # print(loaded_results)

        # # Check if F1 data is within y-limits
        # if all(1<y_value <= 5000 for y_value in loaded_results["A_F1"]):
        #     axs[0, 0].plot(loaded_results["time"], loaded_results["A_F1"][1:], color=color, alpha=0.2)
        #     valid_F1_data.append(loaded_results["A_F1"][1:])

        # # Check if F2 data is within y-limits
        # if all(1<y_value <= 5000 for y_value in loaded_results["A_F2"]):
        #     axs[0, 1].plot(loaded_results["time"], loaded_results["A_F2"][1:], color=color, alpha=0.2)
        #     valid_F2_data.append(loaded_results["A_F2"][1:])

        # # Check if M1 data is within y-limits
        # if all(1<y_value <= 3000 for y_value in loaded_results["A_M1"]):
        #     axs[1, 0].plot(loaded_results["time"], loaded_results["A_M1"][1:], color=color, alpha=0.2)
        #     valid_M1_data.append(loaded_results["A_M1"][1:])

        # # Check if M2 data is within y-limits
        # if all(1<y_value <= 3000 for y_value in loaded_results["A_M2"]):
        #     axs[1, 1].plot(loaded_results["time"], loaded_results["A_M2"][1:], color=color, alpha=0.2)
        #     valid_M2_data.append(loaded_results["A_M2"][1:])

    # # Compute mean curves
    # mean_F1 = np.mean(valid_F1_data, axis=0)
    # mean_F2 = np.mean(valid_F2_data, axis=0)
    # mean_M1 = np.mean(valid_M1_data, axis=0)
    # mean_M2 = np.mean(valid_M2_data, axis=0)

    max_F1 = initialize_problem_from_data("max", "A_F1")[1:]
    min_F1 = initialize_problem_from_data("min", "A_F1")[1:]
    max_F2 = initialize_problem_from_data("max", "A_F2")[1:]
    min_F2 = initialize_problem_from_data("min", "A_F2")[1:]
    max_M1 = initialize_problem_from_data("max", "A_M1")[1:]
    min_M1 = initialize_problem_from_data("min", "A_M1")[1:]
    max_M2 = initialize_problem_from_data("max", "A_M2")[1:]
    min_M2 = initialize_problem_from_data("min", "A_M2")[1:]

    # Plot mean curves with alpha=1 and increased linewidth
    axs[0, 0].plot(loaded_results["time"], mean_F1, label = 'Mean $A_{F1}$',color='black', alpha=1, linewidth=3)
    axs[0, 1].plot(loaded_results["time"], mean_F2, label = 'Mean $A_{F2}$',color='black', alpha=1, linewidth=3)
    axs[1, 0].plot(loaded_results["time"], mean_M1, label = 'Mean $A_{M1}$', color='black', alpha=1, linewidth=3)
    axs[1, 1].plot(loaded_results["time"], mean_M2, label = 'Mean $A_{M2}$', color='black', alpha=1, linewidth=3)

    # Plot maximum and minimum curves
    axs[0, 0].plot(loaded_results["time"], max_F1, label='Max $A_{F1}$', color='blue', alpha=1, linewidth=2)
    axs[0, 0].plot(loaded_results["time"], min_F1, label='Min $A_{F1}$', color='green', alpha=1, linewidth=2)
    axs[0, 1].plot(loaded_results["time"], max_F2, label='Max $A_{F2}$', color='blue', alpha=1, linewidth=2)
    axs[0, 1].plot(loaded_results["time"], min_F2, label='Min $A_{F2}$', color='green', alpha=1, linewidth=2)
    axs[1, 0].plot(loaded_results["time"], max_M1, label='Max $A_{M1}$', color='blue', alpha=1, linewidth=2)
    axs[1, 0].plot(loaded_results["time"], min_M1, label='Min $A_{M1}$', color='green', alpha=1, linewidth=2)
    axs[1, 1].plot(loaded_results["time"], max_M2, label='Max $A_{M2}$', color='blue', alpha=1, linewidth=2)
    axs[1, 1].plot(loaded_results["time"], min_M2, label='Min $A_{M2}$', color='green', alpha=1, linewidth=2)
    
    # Set titles and y-limits for subplots
    # axs[0, 0].set_title('A_F1')
    axs[0, 0].set_ylim(0, 5000)
    axs[0, 0].legend(loc = 'upper center')
    axs[0, 0].set_ylabel('Cell count')
    # axs[0, 1].set_title('A_F2')
    axs[0, 1].set_ylim(0, 5000)
    axs[0, 1].legend(loc = 'upper center')
    axs[0, 1].yaxis.tick_right()
    # axs[1, 0].set_title('A_M1')
    axs[1, 0].set_ylim(0, 3000)
    axs[1, 0].legend(loc = 'upper center')
    axs[1, 0].set_ylabel('Cell count')
    axs[1, 0].set_xlabel('Time(Days)')
    # axs[1, 1].set_title('A_M2')
    axs[1, 1].set_ylim(0, 3000)
    axs[1, 1].legend(loc = 'upper center')
    axs[1, 1].set_xlabel('Time(Days)')
    axs[1, 1].yaxis.tick_right()

    # Save the figure as PNG with specified DPI
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'output_figure_F_c2.png'), dpi=300)



def plot_exp3(nr_exp, out):

    # Load results from all experiments and plot
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    plasma_colormap = matplotlib.cm.get_cmap("viridis")
    colors = [plasma_colormap(x) for x in np.linspace(0.8, 0.15, num=nr_exp)]
    colors = [matplotlib.colors.to_hex(color) for color in colors]


    # Lists to store valid data for computing mean curve
    valid_A_Malpha1_data = []
    valid_A_Malpha2_data = []
    valid_A_MC1_data = []
    valid_A_MC2_data = []

    # Loop through all experiments
    for experiment_to_load, color in zip(range(nr_exp), colors):
        # Load results from the specified experiment
        with open(os.path.join(out, f"experiment_{experiment_to_load}_results.pkl"), "rb") as f:
            loaded_results = pickle.load(f)
        # Check if A_Malpha1 data is within y-limits

        if all(0<y_value <= 1500 for y_value in loaded_results["A_Malpha1"]):
            axs[0, 0].plot(loaded_results["time"], loaded_results["A_Malpha1"][1:], color=color, alpha=0.2)
            valid_A_Malpha1_data.append(loaded_results["A_Malpha1"][1:])

        # Check if A_Malpha2 data is within y-limits
        if all(0<y_value <= 1500 for y_value in loaded_results["A_Malpha2"]):
            axs[0, 1].plot(loaded_results["time"], loaded_results["A_Malpha2"][1:], color=color, alpha=0.2)
            valid_A_Malpha2_data.append(loaded_results["A_Malpha2"][1:])

        # Check if A_MC1 data is within y-limits
        if all(0<y_value <= 5000 for y_value in loaded_results["A_MC1"]):
            axs[1, 0].plot(loaded_results["time"], loaded_results["A_MC1"][1:], color=color, alpha=0.2)
            valid_A_MC1_data.append(loaded_results["A_MC1"][1:])

        # Check if A_MC2 data is within y-limits
        if all(0<y_value <= 5000 for y_value in loaded_results["A_MC2"]):
            axs[1, 1].plot(loaded_results["time"], loaded_results["A_MC2"][1:], color=color, alpha=0.2)
            valid_A_MC2_data.append(loaded_results["A_MC2"][1:])
    # Example usage inside plot_exp1 function with ylim_range specified as (0, 1) for A_Malpha1
    ylim_range_A_Malpha1 = (0, 1)

    
    # Compute mean curves
    mean_A_Malpha1 = np.mean(valid_A_Malpha1_data, axis=0)
    mean_A_Malpha2 = np.mean(valid_A_Malpha2_data, axis=0)
    mean_A_MC1 = np.mean(valid_A_MC1_data, axis=0)
    mean_A_MC2 = np.mean(valid_A_MC2_data, axis=0)

    max_A_Malpha1 = initialize_problem_from_data("max", "A_Malpha1")[1:]
    min_A_Malpha1 = initialize_problem_from_data("min", "A_Malpha1")[1:]
    max_A_Malpha2 = initialize_problem_from_data("max", "A_Malpha2")[1:]
    min_A_Malpha2 = initialize_problem_from_data("min", "A_Malpha2")[1:]
    max_A_MC1 = initialize_problem_from_data("max", "A_MC1")[1:]
    min_A_MC1 = initialize_problem_from_data("min", "A_MC1")[1:]
    max_A_MC2 = initialize_problem_from_data("max", "A_MC2")[1:]
    min_A_MC2 = initialize_problem_from_data("min", "A_MC2")[1:]


    # Plot mean curves with alpha=1 and increased linewidth
    axs[0, 0].plot(loaded_results["time"], mean_A_Malpha1, label = 'Mean $A_{Malpha1}$',color='black', alpha=1, linewidth=3)
    axs[0, 1].plot(loaded_results["time"], mean_A_Malpha2, label = 'Mean $A_{Malpha2}$',color='black', alpha=1, linewidth=3)
    axs[1, 0].plot(loaded_results["time"], mean_A_MC1, label = 'Mean $A_{MC1}$', color='black', alpha=1, linewidth=3)
    axs[1, 1].plot(loaded_results["time"], mean_A_MC2, label = 'Mean $A_{MC2}$', color='black', alpha=1, linewidth=3)


    # Plot maximum and minimum curves
    axs[0, 0].plot(loaded_results["time"], max_A_Malpha1, label='Max $A_{Malpha1}$', color='blue', alpha=1, linewidth=2)
    axs[0, 0].plot(loaded_results["time"], min_A_Malpha1, label='Min $A_{Malpha1}$', color='green', alpha=1, linewidth=2)
    axs[0, 1].plot(loaded_results["time"], max_A_Malpha2, label='Max $A_{Malpha2}$', color='blue', alpha=1, linewidth=2)
    axs[0, 1].plot(loaded_results["time"], min_A_Malpha2, label='Min $A_{Malpha2}$', color='green', alpha=1, linewidth=2)
    axs[1, 0].plot(loaded_results["time"], max_A_MC1, label='Max $A_{MC1}$', color='blue', alpha=1, linewidth=2)
    axs[1, 0].plot(loaded_results["time"], min_A_MC1, label='Min $A_{MC1}$', color='green', alpha=1, linewidth=2)
    axs[1, 1].plot(loaded_results["time"], max_A_MC2, label='Max $A_{MC2}$', color='blue', alpha=1, linewidth=2)
    axs[1, 1].plot(loaded_results["time"], min_A_MC2, label='Min $A_{MC2}$', color='green', alpha=1, linewidth=2)

    # Set titles and y-limits for subplots
    # axs[0, 0].set_title('A_Malpha1')
    axs[0, 0].set_ylim(0, 1500)
    axs[0, 0].legend(loc = 'upper center')
    axs[0, 0].set_ylabel('Cell count')
    # axs[0, 1].set_title('A_Malpha2')
    axs[0, 1].set_ylim(0, 1500)
    axs[0, 1].legend(loc = 'upper center')
    axs[0, 1].yaxis.tick_right()
    # axs[1, 0].set_title('A_MC1')
    axs[1, 0].set_ylim(0, 5000)
    axs[1, 0].legend(loc = 'upper center')
    axs[1, 0].set_xlabel('Time(Days)')
    axs[1, 0].set_ylabel('Cell count')
    # axs[1, 1].set_title('A_MC2')
    axs[1, 1].set_ylim(0, 5000)
    axs[1, 1].legend(loc = 'upper center')
    axs[1, 1].yaxis.tick_right()
    axs[1, 1].set_xlabel('Time(Days)')

    # Save the figure as PNG with specified DPI
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'output_figure_AMalpha_A_MC_c3.png'), dpi=300)



def plot_exp4(nr_exp, out):

    # Load results from all experiments and plot
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    plasma_colormap = matplotlib.cm.get_cmap("Reds")
    colors = [plasma_colormap(x) for x in np.linspace(0.8, 0.15, num=nr_exp)]
    colors = [matplotlib.colors.to_hex(color) for color in colors]


    # Lists to store valid data for computing mean curve
    valid_I1_data = []
    valid_I2_data = []
    valid_beta1_data = []
    valid_beta2_data = []

    # Loop through all experiments
    for experiment_to_load, color in zip(range(nr_exp), colors):
        # Load results from the specified experiment
        with open(os.path.join(out, f"experiment_{experiment_to_load}_results.pkl"), "rb") as f:
            loaded_results = pickle.load(f)
        # Check if I1 data is within y-limits
        if all(0<y_value <= 10**(-5) for y_value in loaded_results["I1"]):
            axs[0, 0].plot(loaded_results["time"], loaded_results["I1"][1:], color=color, alpha=0.2)
            valid_I1_data.append(loaded_results["I1"][1:])

        # Check if I2 data is within y-limits
        if all(0<y_value <= 10**(-5) for y_value in loaded_results["I2"]):
            axs[0, 1].plot(loaded_results["time"], loaded_results["I2"][1:], color=color, alpha=0.2)
            valid_I2_data.append(loaded_results["I2"][1:])

        # Check if beta1 data is within y-limits
        if all(0<y_value <= 10**(-5) for y_value in loaded_results["beta1"]):
            axs[1, 0].plot(loaded_results["time"], loaded_results["beta1"][1:], color=color, alpha=0.2)
            valid_beta1_data.append(loaded_results["beta1"][1:])

        # Check if beta2 data is within y-limits
        if all(0<y_value <= 10**(-5) for y_value in loaded_results["beta2"]):
            axs[1, 1].plot(loaded_results["time"], loaded_results["beta2"][1:], color=color, alpha=0.2)
            valid_beta2_data.append(loaded_results["beta2"][1:])

    mean_I1 = np.mean(valid_I1_data, axis=0)
    mean_I2 = np.mean(valid_I2_data, axis=0)
    mean_beta1 = np.mean(valid_beta1_data, axis=0)
    mean_beta2 = np.mean(valid_beta2_data, axis=0)

    max_I1 = initialize_problem_from_data("max", "I1")[1:]
    min_I1 = initialize_problem_from_data("min", "I1")[1:]
    max_I2 = initialize_problem_from_data("max", "I2")[1:]
    min_I2 = initialize_problem_from_data("min", "I2")[1:]
    max_beta1 = initialize_problem_from_data("max", "beta1")[1:]
    min_beta1 = initialize_problem_from_data("min", "beta1")[1:]
    max_beta2 = initialize_problem_from_data("max", "beta2")[1:]
    min_beta2 = initialize_problem_from_data("min", "beta2")[1:]

    # Plot mean curves with alpha=1 and increased linewidth
    axs[0, 0].plot(loaded_results["time"], mean_I1, label = 'Mean $I1$',color='black', alpha=1, linewidth=3)
    axs[0, 1].plot(loaded_results["time"], mean_I2, label = 'Mean $I2$',color='black', alpha=1, linewidth=3)
    axs[1, 0].plot(loaded_results["time"], mean_beta1, label = 'Mean beta1', color='black', alpha=1, linewidth=3)
    axs[1, 1].plot(loaded_results["time"], mean_beta2, label = 'Mean beta2', color='black', alpha=1, linewidth=3)


    # Plot maximum and minimum curves
    axs[0, 0].plot(loaded_results["time"], max_I1, label='Max $I1$', color='blue', alpha=1, linewidth=2)
    axs[0, 0].plot(loaded_results["time"], min_I1, label='Min $I1$', color='green', alpha=1, linewidth=2)
    axs[0, 1].plot(loaded_results["time"], max_I2, label='Max $I2$', color='blue', alpha=1, linewidth=2)
    axs[0, 1].plot(loaded_results["time"], min_I2, label='Min $I2$', color='green', alpha=1, linewidth=2)
    axs[1, 0].plot(loaded_results["time"], max_beta1, label='Max beta1', color='blue', alpha=1, linewidth=2)
    axs[1, 0].plot(loaded_results["time"], min_beta1, label='Min beta1', color='green', alpha=1, linewidth=2)
    axs[1, 1].plot(loaded_results["time"], max_beta2, label='Max beta2', color='blue', alpha=1, linewidth=2)
    axs[1, 1].plot(loaded_results["time"], min_beta2, label='Min beta2', color='green', alpha=1, linewidth=2)

    # Set titles and y-limits for subplots
    # axs[0, 0].set_title('I1')
    axs[0, 0].set_ylim(0, 10**(-5))
    axs[0, 0].legend(loc = 'upper center')
    axs[0, 0].set_ylabel('Cytokine concentration')
    # axs[0, 1].set_title('I2')
    axs[0, 1].set_ylim(0, 10**(-5))
    axs[0, 1].legend(loc = 'upper center')
    axs[0, 1].yaxis.tick_right()
    # axs[1, 0].set_title('beta1')
    axs[1, 0].set_ylim(0, 10**(-5))
    axs[1, 0].legend(loc = 'upper center')
    axs[1, 0].set_xlabel('Time(Days)')
    axs[1, 0].set_ylabel('Cytokine concentration')
    # axs[1, 1].set_title('beta2')
    axs[1, 1].set_ylim(0, 10**(-5))
    axs[1, 1].legend(loc = 'upper center')
    axs[1, 1].yaxis.tick_right()
    axs[1, 1].set_xlabel('Time(Days)')

    # Save the figure as PNG with specified DPI
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'output_figure_il8_beta_c4.png'), dpi=300)


# Define a function to analyze the experiment results
def analyze_results(results_folder, variable_of_interest, lower_bound, upper_bound):
    # Initialize variables to store best and worst results
    best_result = None
    worst_result = None
    best_variable_value = float('-inf')  # Initialize with negative infinity
    worst_variable_value = float('inf')  # Initialize with positive infinity

    # Loop through the saved experiment results
    for filename in os.listdir(results_folder):
        if filename.endswith(".pkl"):
            with open(os.path.join(results_folder, filename), "rb") as f:
                results = pickle.load(f)
                last_variable_value = results[variable_of_interest][-1]

                # Check if the last variable value is within the specified bounds
                if lower_bound <= last_variable_value <= upper_bound:
                    # Check if the last variable value is the best so far
                    if last_variable_value > best_variable_value:
                        best_variable_value = last_variable_value
                        best_result = {
                            "variable_value": best_variable_value,
                            "initial_values": results["initial_values1"].copy(),
                        }
                    
                    # Check if the last variable value is the worst so far
                    if last_variable_value < worst_variable_value:
                        worst_variable_value = last_variable_value
                        worst_result = {
                            "variable_value": worst_variable_value,
                            "initial_values": results["initial_values1"].copy(),
                        }
    print("Best and worst results achieved for {}!".format(variable_of_interest))    
    print("Outputting results for {}...".format(variable_of_interest))
    with open(os.path.join(output_folder4, f"best_initvalprob_{variable_of_interest}.pkl"), "wb") as f:
            pickle.dump(best_result, f)
    print("Best results outputed!")
    with open(os.path.join(output_folder4, f"worst_initvalprob_{variable_of_interest}.pkl"), "wb") as f:
            pickle.dump(worst_result, f)
    print("Worst results outputed!")
    return best_result, worst_result


results_folder = output_folder5
# perform_exp(nr_exp=1000, output_folder=results_folder)
# get_all_best_within_range(results_folder)

# print(initialize_problem_from_data("best", "CIII1"))
# plot_exp1(2, results_folder)
plot_exp2(2, results_folder)
# plot_exp3(2, results_folder)
# plot_exp4(1000, results_folder)

# with open(os.path.join(output_folder4, f"best_initvalprob_CIII1.pkl"), "rb") as f:
#     loaded_results = pickle.load(f)
# print(loaded_results)
# initial_values = loaded_results['initial_values'] 