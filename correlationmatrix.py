import pandas as pd
import numpy as np
from params import initial_parameters
from param_ranges import p_ranges
from itertools import product
from scipy.spatial.distance import mahalanobis
import glob
import json
import os
from scipy.cluster.hierarchy import dendrogram, linkage
from main import A_MII1_func,A_MII2_func,A_Malpha_func, scenario1_equations, scenario2_equations
import itertools
from sympy import symbols, diff, cos, exp

def run_simulation(parameters):
    # Time parameters
    weeks = 30
    n_days_in_week = 7
    t_max =  weeks * n_days_in_week # Maximum simulation time(weeks)
    dt = weeks/t_max # Time step

    # Forward Euler method
    timesteps = int(t_max / dt)
    time = np.linspace(0, t_max, timesteps)


    # Initialize arrays for results
    A_MII1 = [parameters['A_MII0']]
    I1 = [parameters['I0']]
    beta1 = [parameters['beta0']]
    A_MC1 = [parameters['A_MC0']]
    A_F1 = [parameters['A_F0']]
    A_M1 = [parameters['A_M0']]
    A_Malpha1 = [parameters['A_Malpha0']]
    CIII1 = [parameters['CIII0']]
    CI1 = [parameters['CI0']]

    # Initialize arrays for results
    A_MII2 = [parameters['A_MII0']]
    I2 = [parameters['I0']]
    beta2 = [parameters['beta0']]
    A_MC2 = [parameters['A_MC0']]
    A_F2 = [parameters['A_F0']]
    A_M2 = [parameters['A_M0']]
    A_Malpha2 = [parameters['A_Malpha0']]
    CIII2 = [parameters['CIII0']]
    CI2 = [parameters['CI0']]

    # Perform simulation for both scenarios using forward Euler method
    for i in range(1, timesteps + 1):
        t = i * dt
        
        # Scenario 1
        A_MII_next, I_next, beta_next, A_MC_next, A_F_next, A_M_next, A_Malpha_next, CIII_next, CI_next = \
            scenario1_equations(A_MII1[-1], I1[-1], beta1[-1], A_MC1[-1], A_F1[-1], A_M1[-1], A_Malpha1[-1], CIII1[-1], CI1[-1], parameters, dt, t)
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
            scenario2_equations(A_MII2[-1], I2[-1], beta2[-1], A_MC2[-1], A_F2[-1], A_M2[-1], A_Malpha2[-1], CIII2[-1], CI2[-1], parameters, dt, t)
        A_MII2.append(A_MII_next)
        I2.append(I_next)
        beta2.append(beta_next)
        A_MC2.append(A_MC_next)
        A_F2.append(A_F_next)
        A_M2.append(A_M_next)
        A_Malpha2.append(A_Malpha_next)
        CIII2.append(CIII_next)
        CI2.append(CI_next)
    return A_F1[1:], A_M1[1:], CIII1[1:], CI1[1:], A_F2[1:], A_M2[1:], CIII2[1:], CI2[1:]

def set_parameters(params):
    """
    Set simulation parameters based on the input combination.
    
    Parameters:
        params (tuple): A tuple containing parameter values in the order specified in the initial_parameters function.
    """
    parameters['k1'] = params[0]
    parameters['k2'] = params[1]
    parameters['k3'] = params[2]
    parameters['k4'] = params[3]
    parameters['k5'] = params[4]
    parameters['k6'] = params[5]
    parameters['k7'] = params[6]
    parameters['k8'] = params[7]
    parameters['k9'] = params[8]
    parameters['k10'] = params[9]
    parameters['k11'] = params[10]
    parameters['gamma'] = params[11]
    parameters['zeta'] = params[12]
    parameters['f_dillution'] = params[13]
    parameters['lambda1'] = params[14]
    parameters['lambda2'] = params[15]
    parameters['lambda3'] = params[16]
    parameters['lambda4'] = params[17]
    parameters['rho1'] = params[18]
    parameters['rho2'] = params[19]
    parameters['rho3'] = params[20]
    parameters['mu1'] = params[21]
    parameters['mu2'] = params[22]
    parameters['mu3'] = params[23]
    parameters['mu4'] = params[24]
    parameters['mu5'] = params[25]
    parameters['mu6'] = params[26]
    parameters['mu7'] = params[27]
    parameters['mu8'] = params[28]
    parameters['upsilon1'] = params[29]
    parameters['upsilon2'] = params[30]
    parameters['upsilon3'] = params[31]
    parameters['upsilon4'] = params[32]
    parameters['omega1'] = params[33]
    parameters['omega2'] = params[34]
    parameters['omega3'] = params[35]
    parameters['A_MII0'] = params[36]
    parameters['I0'] = params[37]
    parameters['beta0'] = params[38]
    parameters['A_MC0'] = params[39]
    parameters['A_F0'] = params[40]
    parameters['A_M0'] = params[41]
    parameters['A_Malpha0'] = params[42]
    parameters['CIII0'] = params[43]
    parameters['CI0'] = params[44]

from sympy import symbols, exp


# Define symbols for variables
t = symbols('t')
A_MII, I, beta, A_MC, A_F, A_M, A_Malpha, CIII, CI = symbols('A_MII I beta A_MC A_F A_M A_Malpha CIII CI')

initial_parameters = initial_parameters()
# Define equations
f1 = initial_parameters['f_dillution'] * initial_parameters['k1'] * exp(-initial_parameters['mu1'] * t)
f2 = initial_parameters['f_dillution'] * (-initial_parameters['k2'] * initial_parameters['upsilon1'] * exp(-initial_parameters['upsilon1'] * t) - initial_parameters['mu2'])
f3 = initial_parameters['f_dillution'] * (initial_parameters['k3'] * initial_parameters['upsilon2'] * exp(initial_parameters['upsilon2'] * t) + initial_parameters['k4'] * initial_parameters['gamma'] * A_MII + initial_parameters['k5'] * initial_parameters['gamma'] * A_MC - initial_parameters['mu3'])
f4 = initial_parameters['f_dillution'] * (initial_parameters['lambda3'] * initial_parameters['zeta'] * I - initial_parameters['mu4'])
f5 = initial_parameters['f_dillution'] * (-initial_parameters['lambda1'] * initial_parameters['zeta'] * beta - initial_parameters['rho1'] - initial_parameters['mu5'])
f6 = 0  # Partial derivative of A_M with respect to A_MII is 0
f7 = 0  # Partial derivative of A_Malpha with respect to A_Malpha is 0
f8 = 0  # Partial derivative of CIII with respect to CIII is 0
f9 = 0  # Partial derivative of CI with respect to CI is 0

# Compute partial derivatives
partial_A_MII = f1.diff(A_MII)
partial_I = f2.diff(I)
partial_beta = f3.diff(beta)
partial_A_MC = f4.diff(A_MC)
partial_A_F = f5.diff(A_F)
partial_A_M = f6  # Partial derivative of A_M with respect to A_M is 0
partial_A_Malpha = f7  # Partial derivative of A_Malpha with respect to A_Malpha is 0
partial_CIII = f8  # Partial derivative of CIII with respect to CIII is 0
partial_CI = f9  # Partial derivative of CI with respect to CI is 0

# Construct Jacobian matrix
Jacobian_matrix = [
    [partial_A_MII, partial_I, partial_beta, partial_A_MC, partial_A_F, partial_A_M, partial_A_Malpha, partial_CIII, partial_CI]
]

# Print the Jacobian matrix and individual partial derivatives
print("Jacobian matrix:")
print(Jacobian_matrix)

# Print individual partial derivatives for inspection
print("Partial derivative of A_MII:", partial_A_MII)
print("Partial derivative of I:", partial_I)
print("Partial derivative of beta:", partial_beta)
print("Partial derivative of A_MC:", partial_A_MC)
print("Partial derivative of A_F:", partial_A_F)
print("Partial derivative of A_M:", partial_A_M)
print("Partial derivative of A_Malpha:", partial_A_Malpha)
print("Partial derivative of CIII:", partial_CIII)
print("Partial derivative of CI:", partial_CI)
