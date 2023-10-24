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


ranges_for_production = (0.001, 5)
ranges_for_upsi = (0.001, 0.1)
ranges_for_omega = (0, 2 * np.pi)
ranges_for_gamma = (1.0 * 10**-7, 1.0 * 10**-5)
ranges_for_zeta = (1.0 * 10**-7, 1.0 * 10**-1)
ranges_for_mu = (0.0001, 1)
ranges_for_lambdas = ranges_for_mu
ranges_for_rhos = ranges_for_mu

# Define parameter ranges for sensitivity analysis
p_ranges = {
  'k1': ranges_for_production,
  'k2': ranges_for_production,
  'k3': ranges_for_production,
  'k4': ranges_for_production,
  'k5': ranges_for_production,
  'k6': ranges_for_production,
  'k7': ranges_for_production,
  'k8': ranges_for_production,
  'k9': ranges_for_production,
  'k10': ranges_for_production,
  'k11': ranges_for_production,
  'upsilon1': (-1, 0),
  'upsilon2': (-1, 1),
  'upsilon3': ranges_for_upsi,
  'upsilon4': ranges_for_upsi,
  'lambda1': ranges_for_lambdas,
  'lambda2': ranges_for_lambdas,
  'lambda3': ranges_for_lambdas,
  'lambda4': ranges_for_lambdas,
  'rho1': ranges_for_rhos,
  'rho2': ranges_for_rhos,
  'rho3': ranges_for_rhos,
  'mu1': ranges_for_mu,
  'mu2': ranges_for_mu,
  'mu3': ranges_for_mu,
  'mu4': ranges_for_mu,
  'mu5': ranges_for_mu,
  'mu6': ranges_for_mu,
  'mu7': ranges_for_mu,
  'mu8': ranges_for_mu,
  'A_MII0': (1000, 3000),
  'I0': (10**-9, 10**-6),
  'beta0': (10**-9, 10**-6),
  'A_MC0': (0, 500),
  'A_F0': (0, 500),
  'A_M0': (0, 200),
  'A_Malpha0':(0, 0.000001),
   'CI0':(0, 0.000001),
   'CIII0':(0, 0.000001)
}
