import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
# Load parameter ranges and initial parameters
from param_ranges import parameter_ranges  # Assuming this file contains parameter_ranges dictionary
from params import defined_params, initial_parameters
from main import scenario1_equations, scenario2_equations  # Assuming this file contains scenario1_equations and scenario2_equations functions

def quadratic_curve(t, a, b, c):
    # Quadratic curve function
    return a * t**2 + b * t + c

def cell_dynamics_curve(time, A_MC0, A_F0, A_MII0, time_weeks):
    # Quantities for each cell type
    A_MC = np.array([1.0, 0.8, 0.6, 0.3])  # Mast Cells data points
    A_F = np.array([0.5, 0.8, 0.5, 0.4])  # Fibroblast data points
    A_MII = np.array([1.0, 0.8, 0.6, 0.4])  # MII Macrophages data points

    # Fit quadratic curves to the given data
    popt_mast_cells, _ = curve_fit(quadratic_curve, time_weeks, A_MC * A_MC0)
    popt_fibroblasts, _ = curve_fit(quadratic_curve, time_weeks, A_F * A_F0)
    popt_macrophages, _ = curve_fit(quadratic_curve, time_weeks, A_MII * A_MII0)

    # Generate cell dynamics using the fitted quadratic curves
    A_MII1 = quadratic_curve(time, *popt_macrophages)
    A_MC1 = quadratic_curve(time, *popt_mast_cells)
    A_F1 = quadratic_curve(time, *popt_fibroblasts)

    return np.array([A_MII1, A_MC1, A_F1])

def simulate(parameters, scenario_nr):
    # Unpack parameters
    A_MII0 = parameters['A_MII0']
    I0 = parameters['I0']
    beta0 = parameters['beta0']
    A_MC0 = parameters['A_MC0']
    A_F0 = parameters['A_F0']
    A_M0 = parameters['A_M0']
    A_Malpha0 = parameters['A_Malpha0']
    CIII0 = parameters['CIII0']
    CI0 = parameters['CI0']
    
    # Time parameters
    weeks = 30
    n_days_in_week = 7
    t_max = weeks * n_days_in_week # Maximum simulation time (weeks)
    dt = weeks / t_max  # Time step

    # Forward Euler method
    timesteps = int(t_max / dt)
    time = np.linspace(0, t_max, timesteps)

    # Initialize arrays for results
    if scenario_nr == "1":
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
                scenario1_equations(A_MII1[-1], I1[-1], beta1[-1], A_MC1[-1], A_F1[-1], A_M1[-1], A_Malpha1[-1], CIII1[-1],
                                    CI1[-1], t)
            A_MII1.append(A_MII_next)
            I1.append(I_next)
            beta1.append(beta_next)
            A_MC1.append(A_MC_next)
            A_F1.append(A_F_next)
            A_M1.append(A_M_next)
            A_Malpha1.append(A_Malpha_next)
            CIII1.append(CIII_next)
            CI1.append(CI_next)
            
        # Ensure all lists have the same length
        min_length = min(len(A_MII1), len(A_MC1), len(A_F1))
        A_MII1 = np.array(A_MII1)[:min_length]
        A_MC1 = np.array(A_MC1)[:min_length]
        A_F1 = np.array(A_F1)[:min_length]

        # Convert lists to NumPy arrays and stack them vertically
        simulated_data = np.vstack((A_MII1, A_MC1, A_F1))

    elif scenario_nr == "2":
        A_MII2 = [A_MII0]
        I2 = [I0]
        beta2 = [beta0]
        A_MC2 = [A_MC0]
        A_F2 = [A_F0]
        A_M2 = [A_M0]
        A_Malpha2 = [A_Malpha0]
        CIII2 = [CIII0]
        CI2 = [CI0]

        for i in range(1, timesteps):
            t = i * dt
            # Scenario 2
            A_MII_next, I_next, beta_next, A_MC_next, A_F_next, A_M_next, A_Malpha_next, CIII_next, CI_next = \
                scenario2_equations(A_MII2[-1], I2[-1], beta2[-1], A_MC2[-1], A_F2[-1], A_M2[-1], A_Malpha2[-1], CIII2[-1],
                                    CI2[-1], t)
            A_MII2.append(A_MII_next)
            I2.append(I_next)
            beta2.append(beta_next)
            A_MC2.append(A_MC_next)
            A_F2.append(A_F_next)
            A_M2.append(A_M_next)
            A_Malpha2.append(A_Malpha_next)
            CIII2.append(CIII_next)
            CI2.append(CI_next)
            
        # Convert the lists to numpy arrays before returning
        A_MII2 = np.array(A_MII1).reshape(1, -1)
        I2 = np.array(I1).reshape(1, -1)
        beta2 = np.array(beta1).reshape(1, -1)
        A_MC2 = np.array(A_MC1).reshape(1, -1)
        A_F2 = np.array(A_F1).reshape(1, -1)
        A_M2 = np.array(A_M_next).reshape(1, -1)
        A_Malpha2 = np.array(A_Malpha_next).reshape(1, -1)
        CIII2 = np.array(CIII_next).reshape(1, -1)
        CI2 = np.array(CI_next).reshape(1, -1)

        simulated_data = np.concatenate((A_MII2, A_MC2, A_F2), axis=0)

    return np.array(simulated_data)

def compute_gradients(parameters, non_defined_params, observed_data, scenario_nr):
    # Unpack parameters
    A_MII0 = parameters['A_MII0']
    I0 = parameters['I0']
    beta0 = parameters['beta0']
    A_MC0 = parameters['A_MC0']
    A_F0 = parameters['A_F0']
    A_M0 = parameters['A_M0']
    A_Malpha0 = parameters['A_Malpha0']
    CIII0 = parameters['CIII0']
    CI0 = parameters['CI0']

    # Time parameters
    weeks = 30
    n_days_in_week = 7
    t_max = weeks * n_days_in_week  # Maximum simulation time (weeks)
    dt = weeks / t_max  # Time step

    # Forward Euler method
    timesteps = int(t_max / dt)
    time = np.linspace(0, t_max, timesteps)

    # Simulate data using the provided parameters
    simulated_data = simulate(parameters, scenario_nr)
    # print(np.shape(simulated_data))

    # Ensure the dimensions of simulated_data and observed_data match
    num_params = len(non_defined_params)
    assert simulated_data.shape == observed_data.shape, "Dimension mismatch between simulated_data and observed_data"

    # Compute gradients based on the differences between simulated and observed data
    gradients = simulated_data - observed_data

    # Create a dictionary of gradients
    gradient_dict = {param: gradients[:, param_idx] for param_idx, param in enumerate(non_defined_params)}

    return gradient_dict


def calculate_srme(simulated_data, observed_data):
    error = simulated_data - observed_data
    srme = np.sqrt(np.mean(error**2))
    return srme

def gradient_descent(observed_data, initial_parameters, parameter_ranges, learning_rate=0.01, iterations=1000, scenario_nr="1"):
    non_defined_params = ['k3', 'k5', 'k6', 'k9', 'k10', 'lambda2', 'lambda3', 'lambda4', 'rho2', 'rho3', 'mu3', 'mu4', 'mu6', 'upsilon1', 'upsilon2', 'upsilon3', 'upsilon4', 'omega1', 'omega2', 'omega3', 'A_MII0', 'I0', 'beta0', 'A_MC0', 'A_F0', 'A_M0']
    defined_params = ['k1', 'k2', 'k4', 'k7', 'k8', 'k11', 'lambda1', 'rho1', 'mu1', 'mu2', 'mu5', 'mu7', 'mu8', 'gamma1', 'gamma2', 'gamma3', 'gamma4', 'gamma5', 'gamma6', 'gamma7', 'gamma8', 'zeta1', 'zeta2', 'zeta3', 'zeta4', 'zeta5', 'f_dillution', 'A_Malpha0', 'CIII0', 'CI0']

    # Initialize parameters
    parameters = {param: np.random.uniform(parameter_ranges[param][0], parameter_ranges[param][1]) for param in non_defined_params}
    parameters.update({param: initial_parameters[param] for param in defined_params})

    for _ in range(iterations):
        # Perform simulation with current parameters
        simulated_data = simulate(parameters, scenario_nr)

        # Compute SRME with updated parameters
        srme = calculate_srme(simulated_data, observed_data)

        # Print the current SRME for monitoring progress
        print(f'Iteration: {_}, SRME: {srme}')

        # Compute gradients for non-defined parameters using numerical differentiation or analytical methods
        gradients = compute_gradients(parameters, non_defined_params, observed_data, scenario_nr)

        # Update only non-defined parameters using gradients and learning rate
        new_parameters = parameters.copy()
        for param in non_defined_params:
            new_parameters[param] -= learning_rate * gradients[param]
            # Clip the parameter values within the specified ranges
            new_parameters[param] = np.clip(new_parameters[param], parameter_ranges[param][0], parameter_ranges[param][1])

        # Update parameters for the next iteration
        parameters = new_parameters.copy()

    return parameters


# Time parameters
weeks = 30
n_days_in_week = 7
t_max = weeks * n_days_in_week # Maximum simulation time (weeks)
dt = weeks / t_max  # Time step

# Forward Euler method
timesteps = int(t_max / dt)
time = np.linspace(0, t_max, timesteps+1)
time_weeks = np.array([2 * n_days_in_week, 7 * n_days_in_week, 10 * n_days_in_week, 25 * n_days_in_week])

# Call the function to get cell dynamics lists
cell_dynamics = cell_dynamics_curve(time, 1000, 600, 1000, time_weeks)
# initial_parameters = initial_parameters()
# Perform gradient descent optimization
# optimized_parameters = gradient_descent(observed_data=cell_dynamics, initial_parameters = initial_parameters,parameter_ranges=parameter_ranges, learning_rate=0.01, iterations=1000, scenario_nr="1")

# Plot the cell dynamics
plt.figure(figsize=(8, 6))

# Plot Mast Cells dynamics
plt.plot(time, cell_dynamics[1], label='Mast Cells')

# Plot Fibroblasts dynamics
plt.plot(time, cell_dynamics[2], label='Fibroblasts')

# Plot MII Macrophages dynamics
plt.plot(time, cell_dynamics[0], label='MII Macrophages')

plt.xlabel('Time')
plt.ylabel('Cell Quantity')
plt.title('Cell Dynamics Over Time')
plt.legend()
plt.grid(True)
plt.show()