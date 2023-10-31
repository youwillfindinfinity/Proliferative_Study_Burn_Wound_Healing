import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
import json
from param_ranges import boundfunc
import pickle

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

bounds = boundfunc("array")
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
problem = {
    'num_vars': len(initial_parameters),
    'names': list(initial_parameters.keys()),
    'bounds': bounds
}

with open('model_outputs_scenario1.json', 'r') as f:
    outputs1 = np.array(json.load(f))
with open('sampled_params.json', 'r') as f:
    param_values_list = json.load(f)

# Reshape outputs1 to have dimensions (number of samples, number of outputs)
outputgrouping = ['sc125p', 'sc225p', 'sc150p', 'sc250p', 'sc175p', 'sc275p', 'sc1', 'sc2']
varname = ['A_MII', 'I', 'beta', 'A_MC', 'A_F', 'A_M', 'A_Malpha', 'CIII', 'CI']
for i in range(0, 8):
    outputs1_reshaped = outputs1[:, 0, i, 0, :]
    for j in range(0, 9):
        valout = outputs1_reshaped[:, j]
        Sobol_indices_scenario1 = sobol.analyze(problem, valout, print_to_console=True)
        with open('sobol_indices_'+str(outputgrouping[i])+'_'+str(varname[j])+'.pickle', 'wb') as f:
            pickle.dump(Sobol_indices_scenario1, f)

