import numpy as np
import matplotlib.pyplot as plt
from params import *
from scipy.integrate import solve_ivp


def A_MII1_func(k1, mu1, A_MII0, t):
    A_MII1 = np.exp(-mu1 * t) * A_MII0
    return A_MII1 

# a = A_MII1_func(k1, mu1, A_MII0, time)

def A_MII2_func(k1, mu1, A_MII0, omega1, t):

    A_MII2 = A_MII0*np.exp(-mu1 * t) * np.cos(omega1 * t)
    return A_MII2 

def A_Malpha_func(rho2, A_M, A_Malpha0, t):
    A_Malpha = rho2 * A_M * t * A_Malpha0
    return A_Malpha


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

# Time parameters
weeks = 55
n_days_in_week = 7
t_max =  weeks * n_days_in_week # Maximum simulation time(weeks)
# dt = 0.001 # Time step

# Forward Euler method
timesteps = int(t_max/dt)
time = np.linspace(0, t_max, timesteps)
# print(time, len(time))
# print(IM)

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
    
# # Create subplots
# plt.figure(figsize=(12, 6))

# # Subplot 1
# plt.subplot(1, 2, 1)
# # plt.plot(time, np.array(A_MC1[1:]), label="A_MC1", c='blue')
# # plt.plot(time, np.array(A_MC2[1:]), label="A_MC2", c='lightblue')
# plt.plot(time, np.array(A_MC1[1:]), label="A_MC1")
# plt.plot(time, np.array(A_MC2[1:]), label="A_MC2")
# plt.legend()
# plt.title('Subplot 1')
# plt.xlabel('Time')
# plt.ylabel('Values')

# # Subplot 2
# plt.subplot(1, 2, 2)
# # plt.plot(time, np.array(A_F1[1:]), label="F1", c='red')
# # plt.plot(time, np.array(A_F2[1:]), label="F2", c='orange')
# plt.plot(time, np.array(A_F1[1:]), label="F1")
# plt.plot(time, np.array(A_F2[1:]), label="F2")
# plt.legend()
# plt.title('Subplot 2')
# plt.xlabel('Time')
# plt.ylabel('Values')


# # Adjust layout
# plt.tight_layout()

# # Show the plots
# plt.show()

# # Time parameters
# weeks = 30
# n_days_in_week = 7
# t_max =  weeks * n_days_in_week # Maximum simulation time(weeks)
# dt = weeks/t_max # Time step

# # Forward Euler method
# timesteps = int(t_max / dt)
# time = np.linspace(0, t_max, timesteps)

# # Initial conditions
# t_span = [0, time[-1]]  # Specify the time span for integration

# def scenario1(t, y):
#     A_MII, I, beta, A_MC, A_F, A_M, A_Malpha, CIII, CI = y
#     return scenario1_equations(A_MII, I, beta, A_MC, A_F, A_M, A_Malpha, CIII, CI, t)

# def scenario2(t, y):
#     A_MII, I, beta, A_MC, A_F, A_M, A_Malpha, CIII, CI = y
#     return scenario2_equations(A_MII, I, beta, A_MC, A_F, A_M, A_Malpha, CIII, CI, t)

# y0 = [A_MII0, I0, beta0, A_MC0, A_F0, A_M0, A_Malpha0, CIII0, CI0]
# sol_scenario1 = solve_ivp(scenario1, t_span, y0, method='RK45', dense_output=True)
# sol_scenario2 = solve_ivp(scenario2, t_span, y0, method='RK45', dense_output=True)


# y_scenario1 = sol_scenario1.sol(time)
# y_scenario2 = sol_scenario2.sol(time)


# A_MII1, I1, beta1, A_MC1, A_F1, A_M1, A_Malpha1, CIII1, CI1 = y_scenario1
# A_MII2, I2, beta2, A_MC2, A_F2, A_M2, A_Malpha2, CIII2, CI2 = y_scenario1

def plots_fe(scenario_nr):


    # Plotting concentrations of cytokines, proteins, and cell counts for agents separately for each scenario
    plt.figure(figsize=(15, 12))
    if scenario_nr == "1" or scenario_nr == "both":
        # Scenario 1
        plt.subplot(3, 3, 1)
        plt.plot(time, I1[1:], label=r'$IL-8$(Scenario 1)')
        plt.legend()
        plt.title('IL-8 concentration over time')
        plt.xlabel('Time')
        plt.ylabel('Concentration')

        plt.subplot(3, 3, 2)
        plt.plot(time, beta1[1:], label=r'$TGF-\beta1$(Scenario 1)')
        plt.legend()
        plt.title(r'TGF-$\beta1$ concentration over time')
        plt.xlabel('Time')
        plt.ylabel('Concentration')

        plt.subplot(3, 3, 3)
        plt.plot(time, A_MII1[1:], label=r'$A_{MII}$(Scenario 1)')
        plt.legend()
        plt.title(r'$A_{MII}$ cell count over time')
        plt.xlabel('Time')
        plt.ylabel('Count')

        plt.subplot(3, 3, 4)
        plt.plot(time, A_MC1[1:], label=r'$A_{MC}$(Scenario 1)')
        plt.legend()
        plt.title(r'$A_{MC}$ cell count over time')
        plt.xlabel('Time')
        plt.ylabel('Count')

        plt.subplot(3, 3, 5)
        plt.plot(time, A_F1[1:], label=r'$A_{F}$(Scenario 1)')
        plt.legend()
        plt.title(r'$A_{F}$ cell count over time')
        plt.xlabel('Time')
        plt.ylabel('Count')


        plt.subplot(3, 3, 6)
        plt.plot(time, A_M1[1:], label=r'$A_M$(Scenario 1)')
        plt.legend()
        plt.title(r'$A_M$ cell count over time')
        plt.xlabel('Time')
        plt.ylabel('Count')


        plt.subplot(3, 3, 7)
        plt.plot(time, A_Malpha1[1:], label=r'$A_{M\alpha}$(Scenario 1)')
        plt.legend()
        plt.title(r'$A_{M\alpha}$ cell count over time')
        plt.xlabel('Time')
        plt.ylabel('Count')

        plt.subplot(3, 3, 8)
        plt.plot(time, CIII1[1:], label=r'$C_{III}(Scenario 1)$')
        plt.legend()
        plt.title(r'$C_{III}$ concentration over time ')
        plt.xlabel('Time')
        plt.ylabel('Concentration')


        plt.subplot(3, 3, 9)
        plt.plot(time, CI1[1:], label=r'$C_I$(Scenario 1)')
        plt.legend()
        plt.title(r'$C_I$ concentration over time')
        plt.xlabel('Time')
        plt.ylabel('Concentration')

       

        
#     # print(A_MII2)
#     # print(AM)
    if scenario_nr == "2" or scenario_nr == "both":
        # Scenario 2
        plt.subplot(3, 3, 1)
        plt.plot(time, I2[1:], label=r'$IL-8$(Scenario 2)')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Concentration')

        plt.subplot(3, 3, 2)
        plt.plot(time, beta2[1:], label=r'$TGF-\beta1$(Scenario 2)')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Concentration')

        plt.subplot(3, 3, 3)
        plt.plot(time, A_MII2[1:], label=r'$A_{MII}$(Scenario 2)')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Count')

        plt.subplot(3, 3, 4)
        plt.plot(time, A_MC2[1:], label=r'$A_{MC}$(Scenario 2)')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Count')

        plt.subplot(3, 3, 5)
        plt.plot(time, A_F2[1:], label=r'$A_{F}$(Scenario 2)')
        plt.legend()
        plt.title(r'$A_{F}$ cell count over time')
        plt.xlabel('Time')
        plt.ylabel('Count')
        plt.subplot(3, 3, 6)
        plt.plot(time, A_M2[1:], label=r'$A_M$(Scenario 2)')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Count')

        plt.subplot(3, 3, 7)
        plt.plot(time, A_Malpha2[1:], label=r'$A_{M\alpha}$(Scenario 2)')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Count')

        plt.subplot(3, 3, 8)
        plt.plot(time, CIII2[1:], label=r'$C_{III}$(Scenario 2)')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Concentration')

        plt.subplot(3, 3, 9)
        plt.plot(time, CI2[1:], label=r'$C_{I}$(Scenario 2)')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Concentration')

       


    if scenario_nr == "diff":
        # Scenario 1
        plt.subplot(3, 3, 1)
        plt.plot(time, np.asarray(I1[1:]) - np.asarray(I2[1:]) , label=r'$IL-8$(Scenario 1)')
        plt.legend()
        plt.title('IL-8 concentration over time')
        plt.xlabel('Time')
        plt.ylabel('Concentration')

        plt.subplot(3, 3, 2)
        plt.plot(time, np.asarray(beta1[1:]) - np.asarray(beta2[1:]), label=r'$TGF-\beta1$(Scenario 1)')
        plt.legend()
        plt.title(r'TGF-$\beta1$ concentration over time')
        plt.xlabel('Time')
        plt.ylabel('Concentration')

        plt.subplot(3, 3, 3)
        plt.plot(time, np.asarray(A_MII1[1:]) - np.asarray(A_MII2[1:]), label=r'$A_{MII}$(Scenario 1)')
        plt.legend()
        plt.title(r'$A_{MII}$ cell count over time')
        plt.xlabel('Time')
        plt.ylabel('Count')

        plt.subplot(3, 3, 4)
        plt.plot(time, np.asarray(A_MC1[1:]) - np.asarray(A_MC2[1:]), label=r'$A_{MC}$(Scenario 1)')
        plt.legend()
        plt.title(r'$A_{MC}$ cell count over time')
        plt.xlabel('Time')
        plt.ylabel('Count')

        plt.subplot(3, 3, 5)
        plt.plot(time, np.asarray(A_F1[1:]) - np.asarray(A_F2[1:]), label=r'$A_{F}$(Scenario 1)')
        plt.legend()
        plt.title(r'$A_{F}$ cell count over time')
        plt.xlabel('Time')
        plt.ylabel('Count')


        plt.subplot(3, 3, 6)
        plt.plot(time, np.asarray(A_M1[1:]) - np.asarray(A_M2[1:]), label=r'$A_M$(Scenario 1)')
        plt.legend()
        plt.title(r'$A_M$ cell count over time')
        plt.xlabel('Time')
        plt.ylabel('Count')

        plt.subplot(3, 3, 7)
        plt.plot(time, np.asarray(A_Malpha1[1:]) - np.asarray(A_Malpha2[1:]), label=r'$A_{M\alpha}$(Scenario 1)')
        plt.legend()
        plt.title(r'$A_{M\alpha}$ cell count over time')
        plt.xlabel('Time')
        plt.ylabel('Count')

        plt.subplot(3, 3, 8)
        plt.plot(time, np.asarray(CIII1[1:]) - np.asarray(CIII2[1:]), label=r'$C_{III}(Scenario 1)$')
        plt.legend()
        plt.title(r'$C_{III}$ concentration over time ')
        plt.xlabel('Time')
        plt.ylabel('Concentration')

        plt.subplot(3, 3, 9)
        plt.plot(time, np.asarray(CI1[1:]) - np.asarray(CI2[1:]), label=r'$C_I$(Scenario 1)')
        plt.legend()
        plt.title(r'$C_I$ concentration over time')
        plt.xlabel('Time')
        plt.ylabel('Concentration')

       

       



    plt.tight_layout()
    plt.savefig('simulation_{}'.format(scenario_nr), dpi = 300)
    plt.show()

# plots_fe("both")


# k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, \
# gamma, zeta, f_dillution, lambda1, lambda2, lambda3,\
# lambda4, rho1, rho2, rho3, mu1, mu2, mu3, mu4, mu5, \
# mu6, mu7, mu8, upsilon1, upsilon2, upsilon3, upsilon4, \
# omega1, omega2, omega3, A_MII0, I0, beta0, A_MC0, A_F0, \
# A_M0, A_Malpha0, CIII0, CI0 = 1.439119910008026, 3.3728020021867753, 1.4399999155221739, 4.023532898817432, 1.44, 1.44, 5.0, 5.0, 1.44, 1.44, 1.44, 0.0, -0.04162089475089391, 0.07462911254921226, 0.09999949313304642, 0.14399780357653444, 0.143973473962762, 0.14400000000000002, 0.21711248482614182, 0.14400000000000002, 0.14400000000000002, 0.1479723163153072, 1.0000016895565214, 0.14400000000000002, 0.007131564248382544, 0.061275680627146015, 0.14400000000000002, 0.14400000000000002, 0.14400000000000002, 0.14400000000000002, 1000.0, 1e-06, 1e-06, 0.5002137730367664, 0.6932530570060584, 0.6096239485272107, 1e-06, 1e-06, 4.886424078097683e-11, 5.63958642747979, 5.738349200502124, 6.283185307179586, 1e-05, 1e-05, 0.0625



#1.439119910008026, 3.3728020021867753, 1.4399999155221739, 4.023532898817432, 1.44, 1.44, 5.0, 5.0, 1.44, 1.44, 1.44, 0.0, -0.04162089475089391, 0.07462911254921226, 0.09999949313304642, 0.14399780357653444, 0.143973473962762, 0.14400000000000002, 0.21711248482614182, 0.14400000000000002, 0.14400000000000002, 0.1479723163153072, 1.0000016895565214, 0.14400000000000002, 0.007131564248382544, 0.061275680627146015, 0.14400000000000002, 0.14400000000000002, 0.14400000000000002, 0.14400000000000002, 1000.0, 1e-06, 1e-06, 0.5002137730367664, 0.6932530570060584, 0.6096239485272107, 1e-06, 1e-06, 4.886424078097683e-11, 5.63958642747979, 5.738349200502124, 6.283185307179586, 1e-05, 1e-05, 0.0625

# # Initial conditions
# A_MII0 = 1000
# I0 = 10**(-9) #
# beta0 = 10**(-7) #
# A_MC0 = 1000
# A_F0 = 600
# A_M0 = 50
# A_Malpha0 = 0
# CIII0 = 0
# CI0 = 0
# lambda1, lambda2, k1, mu1, omega1, lambda3, lambda4, mu5, rho1, k2, mu2, k7, k8, rho3, k10 = 25, 25, 10000, 0.1, 0.007, 5, 0.10, 0.000001, 0.01, 0.01, 1, 10, 10, 1, 1


