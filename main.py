
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Define the functions
def f_logistic(K, res_0, r, x):
    return (K) / (1 + ((K - res_0)) / (res_0) * np.exp(-r * x))

def f_powlaw(a, b, x):
    return a * x**b

def f_hill(V_m, n, x, K):
    return (V_m * x**n) / (K**n + x**n)

def f_michmenten(vmax, km, x):
    return (vmax * x) / (km + x)

# Define the system of differential equations
def system(y, t, params):
    M2, F, M, C, MC, IL8, TGFbeta1, VEGF, alphaSMA = y
    kappa_M2, mu_M2, kappa_TGFbeta1M2, mu_TGFbeta1, kappa_IL8MC, mu_IL8, kappa_VEGFM2, kappa_VEGFMC, mu_VEGF, lambda_MCIL8, mu_MC, kappa_FTGFbeta1, sigma_FIL8, sigma_FM2, lambda_FMC, rho_FM, mu_CF, mu_M, kappa_alphaSMA, rho_FM_alphaSMA, mu_alphaSMA, kappa_CF, kappa_CM, lambda_alphaSMAkappa_CM, mu_C = params
    
    dM2dt = (kappa_M2 - mu_M2) * M2
    dFdt = (kappa_TGFbeta1M2 * TGFbeta1 + (sigma_FIL8 * IL8 + sigma_FM2 * M2) * lambda_FMC * MC) - (rho_FM + mu_CF) * F
    dMdt = rho_FM * F - mu_M * M
    dCdt = kappa_CF * F + kappa_CM * M + lambda_alphaSMAkappa_CM * alphaSMA - mu_C * C
    dMCdt = lambda_MCIL8 * IL8 - mu_MC * MC
    dIL8dt = kappa_IL8MC * MC - mu_IL8 * IL8
    dTGFbeta1dt = kappa_IL8MC * IL8 - mu_MC * TGFbeta1
    dVEGFdt = kappa_VEGFM2 * M2 + kappa_VEGFMC * MC - mu_VEGF * VEGF
    dalphaSMAdt = ((kappa_alphaSMA / rho_FM) - mu_alphaSMA) * alphaSMA
    
    return [dM2dt, dFdt, dMdt, dCdt, dMCdt, dIL8dt, dTGFbeta1dt, dVEGFdt, dalphaSMAdt]

# Initial conditions
M2_0 = 1000
F_0 = 250
M_0 = 100
C_0 = 20
MC_0 = 40
IL8_0 = 10**-8
TGFbeta1_0 = 10**-7
VEGF_0 = 10**-9
alphaSMA_0 = 1

# Production/Secretion
kappa_M2 = 0.2
kappa_TGFbeta1M2 = 0.3
kappa_IL8MC = 0.3
kappa_VEGFM2 = 0.2
kappa_VEGFMC = 0.1
kappa_FTGFbeta1 = 0.3
kappa_alphaSMA = 0.3
kappa_CF = 0.5
kappa_CM = 0.2
# Activation
lambda_MCIL8 = 0.2
lambda_FMC = 0.1
lambda_alphaSMAkappa_CM = 0.5
# Inhibition
sigma_FIL8 = 0.3
sigma_FM2 = 0.6
# Transition
rho_FM = 0.1
rho_FM_alphaSMA = 0.3
# Decay
mu_M2 = 0.3
mu_TGFbeta1 = 0.5
mu_IL8 = 0.5
mu_VEGF = 0.2
mu_MC = 0.3
mu_CF = 0.1
mu_M = 0.2
mu_alphaSMA = 0.1
mu_C = 0.3


# Parameters
params = [kappa_M2, mu_M2, kappa_TGFbeta1M2, 
        mu_TGFbeta1, kappa_IL8MC, 
        mu_IL8, kappa_VEGFM2, kappa_VEGFMC, 
        mu_VEGF, lambda_MCIL8, mu_MC,
         kappa_FTGFbeta1, sigma_FIL8, 
         sigma_FM2, lambda_FMC, rho_FM,
          mu_CF, mu_M, kappa_alphaSMA, 
          rho_FM_alphaSMA, mu_alphaSMA, 
          kappa_CF, kappa_CM, 
          lambda_alphaSMAkappa_CM, mu_C]

# Time span and time step
days = 12 * 30 
time_step = 1 * 30   # day time step
num_steps = int(days / time_step)
t = np.linspace(0, days, num_steps + 1)

# Initial conditions
initial_conditions = [M2_0, F_0, M_0, C_0, MC_0, IL8_0, TGFbeta1_0, VEGF_0, alphaSMA_0]

# Euler method to solve ODEs stepwise
results = np.zeros((len(initial_conditions), num_steps + 1))
results[:, 0] = initial_conditions

for i in range(num_steps):
    dydt = system(results[:, i], t[i], params)
    results[:, i + 1] = results[:, i] + np.array(dydt) * time_step

# Plot the results
plt.figure(figsize=(12, 10))

# Plot cells
plt.subplot(3, 1, 1)



plt.plot(t / (1 * 30), results[0], label='M2 Macrophages')
plt.plot(t / (1 * 30), results[1], label='Fibroblasts')
plt.plot(t / (1 * 30), results[2], label='Myofibroblasts')
plt.plot(t / (1 * 30), results[4], label='Mast Cells')
plt.xlabel('Time (Months)')
plt.ylabel('Cell Count')
plt.legend()
plt.title('Cell Dynamics')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # Display integer ticks for months
plt.xticks(range(0, 13))  # Set x-ticks from 0 to 12 months



# plt.plot(t, results[0], label='M2 Macrophages')
# plt.plot(t, results[1], label='Fibroblasts')
# plt.plot(t, results[2], label='Myofibroblasts')
# plt.plot(t, results[4], label='Mast Cells')
# plt.xlabel('Time (days)')
# plt.ylabel('Cell Count')
# plt.legend()
# plt.title('Cell Dynamics')
# plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # Display integer ticks for days
# plt.xticks(range(0, 30))  # Set x-ticks from 0 to 30 days
# plt.xlim(0, 30)
# plt.yscale('log')

# Plot cytokines/proteins
plt.subplot(3, 1, 2)
plt.plot(t / (1 * 30), results[6], label='TGFbeta1')
plt.plot(t / (1 * 30), results[5], label='IL8')
plt.plot(t / (1 * 30), results[7], label='VEGF')
plt.plot(t / (1 * 30), results[3], label='Collagen')
plt.xlabel('Time (Months)')
plt.ylabel('Concentration')
plt.legend()
plt.title('Cytokine/Protein Dynamics')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # Display integer ticks for months
plt.xticks(range(0, 13))  # Set x-ticks from 0 to 12 months
plt.yscale('log')

# Plot alphaSMA
plt.subplot(3, 1, 3)
plt.plot(t / (1 * 30), results[8], label='alphaSMA')
plt.xlabel('Time (Months)')
plt.ylabel('Concentration')
plt.legend()
plt.title('alphaSMA Dynamics')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # Display integer ticks for months
plt.xticks(range(0, 13))  # Set x-ticks from 0 to 12 months
plt.yscale('log')
plt.tight_layout()
plt.show()

