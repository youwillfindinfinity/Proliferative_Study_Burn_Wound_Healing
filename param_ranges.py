import numpy as np


n_values = 10
ranges_for_production = [0.001, 5]
ranges_for_upsi = [0.001, 0.1]
ranges_for_omega = [0, 2 * np.pi]
ranges_for_gamma = [1.0 * 10**-7, 1.0 * 10**-5]
ranges_for_zeta = [1.0 * 10**-7, 1.0 * 10**-1]
ranges_for_mu = [0.0001, 1]
ranges_for_lambdas = ranges_for_mu
ranges_for_rhos = ranges_for_mu

# Define parameter ranges for sensitivity analysis
parameter_ranges = {
  'k1': np.linspace(*ranges_for_production, num = n_values),
  'k2': np.linspace(*ranges_for_production, num = n_values),
  'k3': np.linspace(*ranges_for_production, num = n_values),
  'k4': np.linspace(*ranges_for_production, num = n_values),
  'k5': np.linspace(*ranges_for_production, num = n_values),
  'k6': np.linspace(*ranges_for_production, num = n_values),
  'k7': np.linspace(*ranges_for_production, num = n_values),
  'k8': np.linspace(*ranges_for_production, num = n_values),
  'k9': np.linspace(*ranges_for_production, num = n_values),
  'k10': np.linspace(*ranges_for_production, num = n_values),
  'k11': np.linspace(*ranges_for_production, num = n_values),
  'upsilon1': np.linspace(*ranges_for_upsi, num = n_values),
  'upsilon2': np.linspace(*ranges_for_upsi, num = n_values),
  'upsilon3': np.linspace(*ranges_for_upsi, num = n_values),
  'upsilon4': np.linspace(*ranges_for_upsi, num = n_values),
  'lambda1': np.linspace(*ranges_for_lambdas, num = n_values),
  'lambda2': np.linspace(*ranges_for_lambdas, num = n_values),
  'lambda3': np.linspace(*ranges_for_lambdas, num = n_values),
  'lambda4': np.linspace(*ranges_for_lambdas, num = n_values),
  'rho1': np.linspace(*ranges_for_rhos, num = n_values),
  'rho2': np.linspace(*ranges_for_rhos, num = n_values),
  'rho3': np.linspace(*ranges_for_rhos, num = n_values),
  'mu1': np.linspace(*ranges_for_mu, num = n_values),
  'mu2': np.linspace(*ranges_for_mu, num = n_values),
  'mu3': np.linspace(*ranges_for_mu, num = n_values),
  'mu4': np.linspace(*ranges_for_mu, num = n_values),
  'mu5': np.linspace(*ranges_for_mu, num = n_values),
  'mu6': np.linspace(*ranges_for_mu, num = n_values),
  'mu7': np.linspace(*ranges_for_mu, num = n_values),
  'mu8': np.linspace(*ranges_for_mu, num = n_values),
  'A_MII0': np.linspace(1000, 3000, num = n_values),
  'I0': np.linspace(10**-9, 10**-6, num = n_values),
  'beta0': np.linspace(10**-9, 10**-6, num = n_values),
  'A_MC0': np.linspace(0, 500, num = n_values),
  'A_F0': np.linspace(0, 500, num = n_values),
  'A_M0': np.linspace(0, 200, num = n_values),
  'A_Malpha0':np.linspace(0, 0.000001, num = n_values),
   'CI0':np.linspace(0, 0.000001, num = n_values),
   'CIII0':np.linspace(0, 0.000001, num = n_values)
}

def boundfunc(ftype):
  # Given value of dt
  dt = 1 / (60 * 24)

  # Production Parameters
  k1_min = 1 * 10**(-5) * dt
  k1_max = 5 * 10**(-5) * dt
  k2_min = 1 * 10**(-5) * dt
  k2_max = 5 * 10**(-5) * dt
  k3_min = 1 * 10**(-5) * dt
  k3_max = 9.99 * 10**(-5) * dt
  k4_min = 0.000019 * dt
  k4_max = 0.000099 * dt
  k5_min = 0.00019 * dt
  k5_max = 0.00099 * dt
  k6_min = 30 * dt
  k6_max = 39.9 * dt
  k7_min = 1 * dt
  k7_max = 9.9 * dt
  k8_min = 50 * dt
  k8_max = 59.9 * dt
  k9_min = 30 * dt
  k9_max = 39.9 * dt
  k10_min = 20 * dt
  k10_max = 29.9 * dt

  # Conversion Parameters
  gamma_min = 10**(-6)
  gamma_max = 10**(-3)
  zeta_min = 10**(3)
  zeta_max = 10**(6)

  # Activation Parameters
  lambda1_min = 100 * dt
  lambda1_max = 900 * dt
  lambda2_min = 100 * dt
  lambda2_max = 900 * dt
  lambda3_min = 100 * dt
  lambda3_max = 900 * dt
  lambda4_min = 1*10**(-7) * dt
  lambda4_max = 9*10**(-7) * dt

  # Transition Parameters
  rho1_min = 1 * dt
  rho1_max = 9 * dt
  rho2_min = 1 * dt
  rho2_max = 9 * dt
  rho3_min = 10 * dt
  rho3_max = 19 * dt

  # Decay Parameters
  mu1_min = 1 * dt
  mu1_max = 9 * dt
  mu2_min = 10 * dt
  mu2_max = 19 * dt
  mu3_min = 10 * dt
  mu3_max = 19 * dt
  mu4_min = 10 * dt
  mu4_max = 19 * dt
  mu5_min = 10 * dt
  mu5_max = 19 * dt
  mu6_min = 10 * dt
  mu6_max = 19 * dt
  mu7_min = 1 * dt
  mu7_max = 9 * dt
  mu8_min = 1 * dt
  mu8_max = 9 * dt

  # Sinusoidal Parameters
  upsilon1_min = -0.009
  upsilon1_max = -0.001
  upsilon2_min = -0.9
  upsilon2_max = -0.1
  upsilon3_min = 0.001
  upsilon3_max = 0.009
  upsilon4_min = 0.001
  upsilon4_max = 0.009
  omega1_min = 1 * np.pi * dt
  omega1_max = 9 * np.pi * dt
  omega2_min = 10 * np.pi * dt
  omega2_max = 150 * np.pi * dt
  omega3_min = 10 * np.pi * dt
  omega3_max = 150 * np.pi * dt

  # Initial Conditions
  A_MII0_min = 900
  A_MII0_max = 2100
  I0_min = 0.1 * 10**(-8)
  I0_max = 9.9 * 10**(-8)
  beta0_min = 0.1 * 10**(-7)
  beta0_max = 9.9 * 10**(-7)
  A_MC0_min = 900
  A_MC0_max = 2100
  A_F0_min = 300
  A_F0_max = 900
  A_M0_min = 50
  A_M0_max = 800
  A_Malpha0_min = 0
  A_Malpha0_max = 10
  CIII0_min = 0.001
  CIII0_max = 0.009
  CI0_min = 0.001
  CI0_max = 0.009

  if ftype == "array":
    # Creating bounds list
    bounds = np.array([
        [k1_min, k1_max],
        [k2_min, k2_max],
        [k3_min, k3_max],
        [k4_min, k4_max],
        [k5_min, k5_max],
        [k6_min, k6_max],
        [k7_min, k7_max],
        [k8_min, k8_max],
        [k9_min, k9_max],
        [k10_min, k10_max],
        [gamma_min, gamma_max],
        [zeta_min, zeta_max],
        [lambda1_min, lambda1_max],
        [lambda2_min, lambda2_max],
        [lambda3_min, lambda3_max],
        [lambda4_min, lambda4_max],
        [rho1_min, rho1_max],
        [rho2_min, rho2_max],
        [rho3_min, rho3_max],
        [mu1_min, mu1_max],
        [mu2_min, mu2_max],
        [mu3_min, mu3_max],
        [mu4_min, mu4_max],
        [mu5_min, mu5_max],
        [mu6_min, mu6_max],
        [mu7_min, mu7_max],
        [mu8_min, mu8_max],
        [upsilon1_min, upsilon1_max],
        [upsilon2_min, upsilon2_max],
        [upsilon3_min, upsilon3_max],
        [upsilon4_min, upsilon4_max],
        [omega1_min, omega1_max],
        [omega2_min, omega2_max],
        [omega3_min, omega3_max],
        [A_MII0_min, A_MII0_max],
        [I0_min, I0_max],
        [beta0_min, beta0_max],
        [A_MC0_min, A_MC0_max],
        [A_F0_min, A_F0_max],
        [A_M0_min, A_M0_max],
        [A_Malpha0_min, A_Malpha0_max],
        [CIII0_min, CIII0_max],
        [CI0_min, CI0_max]
    ])
  if ftype == "dict":
    # Creating bounds list
    bounds = {
        'k1':[k1_min, k1_max],
        'k2':[k2_min, k2_max],
        'k3':[k3_min, k3_max],
        'k4':[k4_min, k4_max],
        'k5':[k5_min, k5_max],
        'k6':[k6_min, k6_max],
        'k7':[k7_min, k7_max],
        'k8':[k8_min, k8_max],
        'k9':[k9_min, k9_max],
        'k10':[k10_min, k10_max],
        'gamma':[gamma_min, gamma_max],
        'zeta':[zeta_min, zeta_max],
        'lambda1':[lambda1_min, lambda1_max],
        'lambda2':[lambda2_min, lambda2_max],
        'lambda3':[lambda3_min, lambda3_max],
        'lambda4':[lambda4_min, lambda4_max],
        'rho1':[rho1_min, rho1_max],
        'rho2':[rho2_min, rho2_max],
        'rho3':[rho3_min, rho3_max],
        'mu1':[mu1_min, mu1_max],
        'mu2':[mu2_min, mu2_max],
        'mu3':[mu3_min, mu3_max],
        'mu4':[mu4_min, mu4_max],
        'mu5':[mu5_min, mu5_max],
        'mu6':[mu6_min, mu6_max],
        'mu7':[mu7_min, mu7_max],
        'mu8':[mu8_min, mu8_max],
        'upsilon1':[upsilon1_min, upsilon1_max],
        'upsilon2':[upsilon2_min, upsilon2_max],
        'upsilon3':[upsilon3_min, upsilon3_max],
        'upsilon4':[upsilon4_min, upsilon4_max],
        'omega1':[omega1_min, omega1_max],
        'omega2':[omega2_min, omega2_max],
        'omega3':[omega3_min, omega3_max],
        'A_MII0':[A_MII0_min, A_MII0_max],
        'I0':[I0_min, I0_max],
        'beta0':[beta0_min, beta0_max],
        'A_MC0':[A_MC0_min, A_MC0_max],
        'A_F0':[A_F0_min, A_F0_max],
        'A_M0':[A_M0_min, A_M0_max],
        'A_Malpha0':[A_Malpha0_min, A_Malpha0_max],
        'CIII0':[CIII0_min, CIII0_max],
        'CI0':[CI0_min, CI0_max]
    }
  return bounds
