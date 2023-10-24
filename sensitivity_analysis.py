import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy
import json
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler

from main import scenario1_equations, scenario2_equations

# Define scenarios and parameters 
day_conversion = 60 * 24 #
# Production Parameters 
k1 = 2.34 * 10**-6 # rho2 https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009839
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
f_dillution = 1/16 #

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
mu1 = 0.07 # day-1 mu_AM https://link.springer.com/article/10.1186/1471-2105-14-S6-S7/tables/2
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
# Time parameters
weeks = 30
n_days_in_week = 7
t_max =  weeks * n_days_in_week # Maximum simulation time(weeks)
dt = weeks/t_max # Time step

# Forward Euler method
timesteps = int(t_max / dt)
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


# Calculate parameter correlation matrix for Scenario 1
parameters_scenario1 = np.array([k1, mu1, A_MII0, omega1, f_dillution, k2, upsilon1, k6, gamma, lambda3, zeta,
                                 mu2, k3, upsilon2, k4, mu3, lambda2, lambda1, rho1, mu5, rho1, lambda1, zeta,
                                 lambda4, mu6, rho2, A_Malpha0, k7, k10, rho3, mu7, k9, k8, k11, mu8])

# Calculate parameter correlation matrix for Scenario 2
parameters_scenario2 = np.array([k1, mu1, A_MII0, omega1, f_dillution, k2, upsilon3, omega2, k6, gamma, lambda3, zeta,
                                 mu2, k3, upsilon4, omega3, mu3, lambda2, lambda1, rho1, mu5, rho1, lambda1, zeta,
                                 lambda4, mu6, rho2, A_Malpha0, k7, k10, rho3, mu7, k9, k8, k11, mu8])



# Define parameter names as strings
parameters_scenario1_list = [
    'k1', 'mu1', 'A_MII0', 'omega1', 'f_dillution', 'k2', 'upsilon1', 'k6', 'gamma', 'lambda3', 'zeta',
    'mu2', 'k3', 'upsilon2', 'k4', 'mu3', 'lambda2', 'lambda1', 'rho1', 'mu5', 'rho1', 'lambda1', 'zeta',
    'lambda4', 'mu6', 'rho2', 'A_Malpha0', 'k7', 'k10', 'rho3', 'mu7', 'k9', 'k8', 'k11', 'mu8'
]

parameters_scenario2_list = [
    'k1', 'mu1', 'A_MII0', 'omega1', 'f_dillution', 'k2', 'upsilon3', 'omega2', 'k6', 'gamma', 'lambda3', 'zeta',
    'mu2', 'k3', 'upsilon4', 'omega3', 'mu3', 'lambda2', 'lambda1', 'rho1', 'mu5', 'rho1', 'lambda1', 'zeta',
    'lambda4', 'mu6', 'rho2', 'A_Malpha0', 'k7', 'k10', 'rho3', 'mu7', 'k9', 'k8', 'k11', 'mu8'
]

# Initialize arrays for parameter sensitivity
sensitivity_scenario1 = np.zeros((timesteps, len(parameters_scenario1)))
sensitivity_scenario2 = np.zeros((timesteps, len(parameters_scenario2)))

# Perform simulation for both scenarios using forward Euler method
for i in range(timesteps):
    t = i * dt

    # Scenario 1
    A_MII_next, I_next, beta_next, A_MC_next, A_F_next, A_M_next, A_Malpha_next, CIII_next, CI_next = \
        scenario1_equations(A_MII1[-1], I1[-1], beta1[-1], A_MC1[-1], A_F1[-1], A_M1[-1], A_Malpha1[-1], CIII1[-1], CI1[-1], t)
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
        scenario2_equations(A_MII2[-1], I2[-1], beta2[-1], A_MC2[-1], A_F2[-1], A_M2[-1], A_Malpha2[-1], CIII2[-1], CI2[-1], t)
    A_MII2.append(A_MII_next)
    I2.append(I_next)
    beta2.append(beta_next)
    A_MC2.append(A_MC_next)
    A_F2.append(A_F_next)
    A_M2.append(A_M_next)
    A_Malpha2.append(A_Malpha_next)
    CIII2.append(CIII_next)
    CI2.append(CI_next)

    # Calculate sensitivities for Scenario 1
    for j in range(len(parameters_scenario1)):
        # Perturb the parameter slightly
        perturbed_params = parameters_scenario1.copy()
        perturbed_params[j] *= 1.10  # Increase the parameter by 10%

        # Rerun the simulation with the perturbed parameter
        A_MII_perturbed, I_perturbed, beta_perturbed, A_MC_perturbed, A_F_perturbed, A_M_perturbed, A_Malpha_perturbed, CIII_perturbed, CI_perturbed = \
            scenario1_equations(A_MII1[-1], I1[-1], beta1[-1], A_MC1[-1], A_F1[-1], A_M1[-1], A_Malpha1[-1], CIII1[-1], CI1[-1], t)
        # Calculate sensitivity as the relative change in the output
        sensitivity_scenario1[i, j] = (A_MII_perturbed - A_MII1[-1]) / (A_MII1[-1] + 1e-6)
        # Calculate sensitivity as the relative change in the output
        sensitivity_scenario1[i, j] = (I_perturbed - I1[-1]) / (I1[-1] + 1e-6)
        # Calculate sensitivity as the relative change in the output
        sensitivity_scenario1[i, j] = (beta_perturbed - beta1[-1]) / (beta1[-1] + 1e-6)
        # Calculate sensitivity as the relative change in the output
        sensitivity_scenario1[i, j] = (A_MC_perturbed - A_MC1[-1]) / (A_MC1[-1] + 1e-6)
        # Calculate sensitivity as the relative change in the output
        sensitivity_scenario1[i, j] = (A_F_perturbed - A_F1[-1]) / (A_F1[-1] + 1e-6)
        # Calculate sensitivity as the relative change in the output
        sensitivity_scenario1[i, j] = (A_M_perturbed - A_M1[-1]) / (A_M1[-1] + 1e-6)
        # Calculate sensitivity as the relative change in the output
        sensitivity_scenario1[i, j] = (A_Malpha_perturbed - A_Malpha1[-1]) / (A_Malpha1[-1] + 1e-6)
        # Calculate sensitivity as the relative change in the output
        sensitivity_scenario1[i, j] = (CIII_perturbed - CIII1[-1]) / (CIII1[-1] + 1e-6)
        # Calculate sensitivity as the relative change in the output
        sensitivity_scenario1[i, j] = (CI_perturbed - CI1[-1]) / (CI1[-1] + 1e-6)

    # Calculate sensitivities for Scenario 2
    for j in range(len(parameters_scenario2)):
        # Perturb the parameter slightly
        perturbed_params = parameters_scenario2.copy()
        perturbed_params[j] *= 1.10  # Increase the parameter by 10%
        # Rerun the simulation with the perturbed parameter
        A_MII_perturbed, I_perturbed, beta_perturbed, A_MC_perturbed, A_F_perturbed, A_M_perturbed, A_Malpha_perturbed, CIII_perturbed, CI_perturbed = \
            scenario2_equations(A_MII2[-1], I2[-1], beta2[-1], A_MC2[-1], A_F2[-1], A_M2[-1], A_Malpha2[-1], CIII2[-1], CI2[-1], t)
        # Calculate sensitivity as the relative change in the output
        sensitivity_scenario2[i, j] = (A_MII_perturbed - A_MII2[-1]) / (A_MII2[-1] + 1e-6)
        # Calculate sensitivity as the relative change in the output
        sensitivity_scenario2[i, j] = (I_perturbed - I2[-1]) / (I2[-1] + 1e-6)
        # Calculate sensitivity as the relative change in the output
        sensitivity_scenario2[i, j] = (beta_perturbed - beta2[-1]) / (beta2[-1] + 1e-6)
        # Calculate sensitivity as the relative change in the output
        sensitivity_scenario2[i, j] = (A_MC_perturbed - A_MC2[-1]) / (A_MC2[-1] + 1e-6)
        # Calculate sensitivity as the relative change in the output
        sensitivity_scenario2[i, j] = (A_F_perturbed - A_F2[-1]) / (A_F2[-1] + 1e-6)
        # Calculate sensitivity as the relative change in the output
        sensitivity_scenario2[i, j] = (A_M_perturbed - A_M2[-1]) / (A_M2[-1] + 1e-6)
        # Calculate sensitivity as the relative change in the output
        sensitivity_scenario2[i, j] = (A_Malpha_perturbed - A_Malpha2[-1]) / (A_Malpha2[-1] + 1e-6)
        # Calculate sensitivity as the relative change in the output
        sensitivity_scenario2[i, j] = (CIII_perturbed - CIII2[-1]) / (CIII2[-1] + 1e-6)
        # Calculate sensitivity as the relative change in the output
        sensitivity_scenario2[i, j] = (CI_perturbed - CI2[-1]) / (CI2[-1] + 1e-6)

# Convert time to weeks
time_in_weeks = time / 7

# Set the index of sensitivity dataframes to time in weeks
sensitivity_df_scenario1 = pd.DataFrame(sensitivity_scenario1, columns=parameters_scenario1_list, index=time_in_weeks)
sensitivity_df_scenario2 = pd.DataFrame(sensitivity_scenario2, columns=parameters_scenario2_list, index=time_in_weeks)

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Scale sensitivity dataframes
scaled_sensitivity_df_scenario1 = pd.DataFrame(scaler.fit_transform(sensitivity_df_scenario1), columns=parameters_scenario1_list, index=time_in_weeks)
scaled_sensitivity_df_scenario2 = pd.DataFrame(scaler.fit_transform(sensitivity_df_scenario2), columns=parameters_scenario2_list, index=time_in_weeks)

scaled_sensitivity_df_scenario1.fillna(0, inplace=True)
scaled_sensitivity_df_scenario1.replace([np.inf, -np.inf], 0, inplace=True)
print(scaled_sensitivity_df_scenario1.isna().sum())
print(np.isinf(scaled_sensitivity_df_scenario1).sum())

# print(scaled_sensitivity_df_scenario1)
# Set Seaborn style
sns.set(style="white")

# plt.figure(figsize=(12, 8))
# plt.title('Maximum Parameter Sensitivity for Scenario 1')
# sns.heatmap(data=scaled_sensitivity_df_scenario1.T, annot=True, fmt=".2f", cmap='coolwarm', xticklabels=np.arange(0, weeks + 1), yticklabels=parameters_scenario1_list)
# plt.xlabel('Time (Weeks)')
# plt.ylabel('Parameters')

print(scaled_sensitivity_df_scenario1)
# sns.clustermap(scaled_sensitivity_df_scenario1, metric="euclidean", standard_scale=1, annot=True, fmt=".2f", cmap='coolwarm', xticklabels=np.arange(0, weeks + 1), yticklabels=parameters_scenario1_list)
# plt.show()


# # Plot dendrogram for Scenario 2 based on parameter sensitivity
# plt.figure(figsize=(12, 6))
# plt.title('Dendrogram for Scenario 2 (Parameter Sensitivity)')
# sns.clustermap(data=scaled_sensitivity_df_scenario2, figsize=(12, 6), row_cluster=False, cmap="coolwarm")
# plt.show()
