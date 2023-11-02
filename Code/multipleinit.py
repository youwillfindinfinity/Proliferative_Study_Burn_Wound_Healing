
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import pickle
import os


# Create a directory to store pickle files
output_folder1 = "pickle_plasma"
os.makedirs(output_folder1, exist_ok=True)


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


def perform_exp(nr_exp):

        
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
        'A_MII0': (500, 3500),
        'I0': (1e-9, 5e-7),
        'beta0': (1e-9, 5e-7),
        'A_MC0': (500, 3500),
        'A_F0': (0,2800),
        'A_M0': (0,1000),
        'A_Malpha0': (0,500),
        'CIII0': (0,1),
        'CI0': (0,0.1)
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
        
        with open(os.path.join(output_folder3, f"experiment_{experiment}_results.pkl"), "wb") as f:
            pickle.dump(results, f)
    # return



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

    # Compute max and min curves
    max_CIII1 = np.max(valid_CIII1_data, axis=0)
    max_CIII2 = np.max(valid_CIII2_data, axis=0)
    max_CI1 = np.max(valid_CI1_data, axis=0)
    max_CI2 = np.max(valid_CI2_data, axis=0)

    min_CIII1 = np.min(valid_CIII1_data, axis=0)
    min_CIII2 = np.min(valid_CIII2_data, axis=0)
    min_CI1 = np.min(valid_CI1_data, axis=0)
    min_CI2 = np.min(valid_CI2_data, axis=0)

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
    plt.savefig(os.path.join(out, 'output_figure_C_c2.png'), dpi=300)


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

        # Check if F1 data is within y-limits
        if all(1<y_value <= 5000 for y_value in loaded_results["A_F1"]):
            axs[0, 0].plot(loaded_results["time"], loaded_results["A_F1"][1:], color=color, alpha=0.2)
            valid_F1_data.append(loaded_results["A_F1"][1:])

        # Check if F2 data is within y-limits
        if all(1<y_value <= 5000 for y_value in loaded_results["A_F2"]):
            axs[0, 1].plot(loaded_results["time"], loaded_results["A_F2"][1:], color=color, alpha=0.2)
            valid_F2_data.append(loaded_results["A_F2"][1:])

        # Check if M1 data is within y-limits
        if all(1<y_value <= 3000 for y_value in loaded_results["A_M1"]):
            axs[1, 0].plot(loaded_results["time"], loaded_results["A_M1"][1:], color=color, alpha=0.2)
            valid_M1_data.append(loaded_results["A_M1"][1:])

        # Check if M2 data is within y-limits
        if all(1<y_value <= 3000 for y_value in loaded_results["A_M2"]):
            axs[1, 1].plot(loaded_results["time"], loaded_results["A_M2"][1:], color=color, alpha=0.2)
            valid_M2_data.append(loaded_results["A_M2"][1:])

    # Compute mean curves
    mean_F1 = np.mean(valid_F1_data, axis=0)
    mean_F2 = np.mean(valid_F2_data, axis=0)
    mean_M1 = np.mean(valid_M1_data, axis=0)
    mean_M2 = np.mean(valid_M2_data, axis=0)

    # Compute max and min curves
    max_F1 = np.max(valid_F1_data, axis=0)
    max_F2 = np.max(valid_F2_data, axis=0)
    max_M1 = np.max(valid_M1_data, axis=0)
    max_M2 = np.max(valid_M2_data, axis=0)

    min_F1 = np.min(valid_F1_data, axis=0)
    min_F2 = np.min(valid_F2_data, axis=0)
    min_M1 = np.min(valid_M1_data, axis=0)
    min_M2 = np.min(valid_M2_data, axis=0)

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

