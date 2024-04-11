import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from params import initial_parameters

def A_MII1_func(mu1, A_MII0, t):
    A_MII1 = np.exp(-mu1 * t) * A_MII0
    return A_MII1 


def A_MII2_func(mu1, A_MII0, omega1, t):
    A_MII2 = A_MII0*np.exp(-mu1 * t) * np.cos(omega1 * t)
    return A_MII2 

def A_Malpha_func(rho2, A_M, A_Malpha0, t):
    A_Malpha = rho2 * A_M * t * A_Malpha0
    return A_Malpha



# print(parameters)
# Scenario 1 equations
def scenario1_equations(A_MII, I, beta, A_MC, A_F, A_M, A_Malpha, CIII, CI, parameters, dt, t):
    '''
    Solves step dt for each of the equations in scenario 1
    '''
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
    '''
    Solves step dt for each of the equations in scenario 2
    '''
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

def solver(parameters_dict):
    parameters = parameters_dict
    # Time parameters
    weeks = 30
    n_days_in_week = 7
    t_max =  weeks * n_days_in_week # Maximum simulation time(weeks)
    dt = parameters["dt"]
    # Forward Euler method
    timesteps = int(t_max/dt)
    time = np.linspace(0, t_max, timesteps)

    # Initialize arrays for results
    A_MII1 = [parameters["A_MII0"]]
    I1 = [parameters["I0"]]
    beta1 = [parameters["beta0"]]
    A_MC1 = [parameters["A_MC0"]]
    A_F1 = [parameters["A_F0"]]
    A_M1 = [parameters["A_M0"]]
    A_Malpha1 = [parameters["A_Malpha0"]]
    CIII1 = [parameters["CIII0"]]
    CI1 = [parameters["CI0"]]

    A_MII2 = [parameters["A_MII0"]]
    I2 = [parameters["I0"]]
    beta2 = [parameters["beta0"]]
    A_MC2 = [parameters["A_MC0"]]
    A_F2 = [parameters["A_F0"]]
    A_M2 = [parameters["A_M0"]]
    A_Malpha2 = [parameters["A_Malpha0"]]
    CIII2 = [parameters["CIII0"]]
    CI2 = [parameters["CI0"]]

    # Perform simulation for both scenarios using forward Euler method
    for i in range(1, timesteps + 1):
        t = i * parameters["dt"]
        
        # Scenario 1
        A_MII_next, I_next, beta_next, A_MC_next, A_F_next, A_M_next, A_Malpha_next, CIII_next, CI_next = \
            scenario1_equations(A_MII1[-1], I1[-1], beta1[-1], A_MC1[-1], A_F1[-1], A_M1[-1], A_Malpha1[-1], CIII1[-1], CI1[-1],parameters, parameters["dt"], t)
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
            scenario2_equations(A_MII2[-1], I2[-1], beta2[-1], A_MC2[-1], A_F2[-1], A_M2[-1], A_Malpha2[-1], CIII2[-1], CI2[-1],parameters, parameters["dt"], t)
        A_MII2.append(A_MII_next)
        I2.append(I_next)
        beta2.append(beta_next)
        A_MC2.append(A_MC_next)
        A_F2.append(A_F_next)
        A_M2.append(A_M_next)
        A_Malpha2.append(A_Malpha_next)
        CIII2.append(CIII_next)
        CI2.append(CI_next)
    return time, dt, A_MII1, I1, beta1, A_MC1, A_F1, A_M1, A_Malpha1, CIII1, CI1, A_MII2, I2, beta2, A_MC2, A_F2, A_M2, A_Malpha2, CIII2, CI2



def plots_fe(scenario_nr, parameters_dict):
    '''
    Plots the results from each list
    '''

    time, dt, A_MII1, I1, beta1, A_MC1, A_F1, A_M1, A_Malpha1, CIII1, CI1, A_MII2, I2, beta2, A_MC2, A_F2, A_M2, A_Malpha2, CIII2, CI2 = solver(parameters_dict)
    
    plt.figure(figsize=(12, 6))

    if scenario_nr == "1" or scenario_nr == "both":
        # Scenario 1
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(time, A_MII1[1:], label=r'$A_{MII}$(Sc.1)', c="darkblue")
        ax1.plot(time, A_MC1[1:], label=r'$A_{MC}$(Sc.1)', c="orange")
        ax1.legend(loc="lower left", ncol=2)
        ax1.set_ylabel('Cell count')
        ax1.set_xticks([])  # Turn off x-axis labels

        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(time, A_F1[1:], label=r'$A_{F}$(Sc.1)', c="black")
        ax2.plot(time, A_Malpha1[1:], label=r'$A_{M\alpha}$(Sc.1)', c="lightsteelblue")
        ax2.legend(loc="upper right", ncol=2)
        ax2.set_xticks([])  # Turn off x-axis labels

        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(time, I1[1:], label=r'$IL-8$(Sc.1)', c="red")
        ax3.plot(time, beta1[1:], label=r'$TGF-\beta1$(Sc.1)', c="blue")
        ax3.legend(loc="lower right", ncol=2)
        ax3.set_xlabel('Time(days)')
        ax3.set_ylabel('Chemokine Concentration')

        ax4 = plt.subplot(2, 2, 4)
        ax4.plot(time, CIII1[1:], label=r'$C_{III}(Sc.1)$', c="lightgreen")
        ax4.plot(time, CI1[1:], label=r'$C_I$(Sc.1)', c="green")
        ax4.legend(loc="upper left", ncol=2)
        ax4.set_xlabel('Time(days)')

        # Set y-labels on the right side for subplots on the right
        ax2.yaxis.set_label_position("right")
        ax4.yaxis.set_label_position("right")
        ax2.set_ylabel('Cell count')
        ax4.set_ylabel('Collagen Concentration')

        # Add ticks and y values to the right for subplots on the right
        ax2.yaxis.tick_right()
        ax4.yaxis.tick_right()

    if scenario_nr == "2" or scenario_nr == "both":
        # Scenario 2
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(time, A_MII2[1:], label=r'$A_{MII}$(Sc.2)', c="darkblue", linestyle="dotted")
        ax1.plot(time, A_MC2[1:], label=r'$A_{MC}$(Sc.2)', c="orange", linestyle="dotted")
        ax1.legend(loc="lower left", ncol=2)
        ax1.set_ylabel('Cell count')
        ax1.set_xticks([])  # Turn off x-axis labels

        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(time, A_F2[1:], label=r'$A_{F}$(Sc.2)', c="black", linestyle="dotted")
        ax2.plot(time, A_Malpha2[1:], label=r'$A_{M\alpha}$(Sc.2)', c="lightsteelblue", linestyle="dotted")
        ax2.legend(loc="upper right", ncol=2)
        ax2.set_xticks([])  # Turn off x-axis labels

        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(time, I2[1:], label=r'$IL-8$(Sc.2)', c="red", linestyle="dotted")
        ax3.plot(time, beta2[1:], label=r'$TGF-\beta1$(Sc.2)', c="blue", linestyle="dotted")
        ax3.legend(loc="lower right", ncol=2)
        ax3.set_xlabel('Time(days)')
        ax3.set_ylabel('Chemokine Concentration')

        ax4 = plt.subplot(2, 2, 4)
        ax4.plot(time, CIII2[1:], label=r'$C_{III}(Sc.2)$', c="lightgreen", linestyle="dotted")
        ax4.plot(time, CI2[1:], label=r'$C_{I}$(Sc.2)', c="green", linestyle="dotted")
        ax4.legend(loc="upper left", ncol=2)
        ax4.set_xlabel('Time(days)')

        # Set y-labels on the right side for subplots on the right
        ax2.yaxis.set_label_position("right")
        ax4.yaxis.set_label_position("right")
        ax2.set_ylabel('Cell count')
        ax4.set_ylabel('Collagen Concentration')

        # Add ticks and y values to the right for subplots on the right
        ax2.yaxis.tick_right()
        ax4.yaxis.tick_right()

         # Set y-labels on the right side for subplots on the right
        ax3.yaxis.set_label_position("left")
        ax3.set_ylabel('Chemokine Concentration')

        # Format y-axis labels in scientific notation with specified format
        ax3.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax3.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.tight_layout()
    plt.savefig('simulation_{}'.format(scenario_nr), dpi=300)
    plt.show()

# Call the function with scenario_nr set to "both" or any scenario


# parameters = initial_parameters()
# solver(parameters)
# plots_fe("both", parameters)