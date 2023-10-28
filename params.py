import numpy as np

def defined_params():
	day_conversion = 24 * 60
	defined_dict = {
    'k1': 2.34 * 10**-5,
    'k2': 234 * 10**-5 * day_conversion,
    'k4': 280 * 10**-5 * day_conversion,
    'k7': 50,
    'k8': 30,
    'k11': 2 * 10**(-7) * day_conversion,
    'lambda1': 0.001 * day_conversion,
    'rho1': 0.3,
    'mu1': 0.07,
    'mu2': 7,
    'mu5': 0.1,
    'mu7': 9.7 * 10**(-5) * day_conversion,
    'mu8': 9.7 * 10**(-5) * day_conversion,
    'gamma': 10**(-5),
    'zeta': 10**(-5),
    'A_Malpha0': 0,
    'CIII0': 0,
    'CI0': 0,
    'f_dillution': 1/16
}

	return defined_dict

def initial_parameters():
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
	return parameters

# # Time parameters
# weeks = 30
# n_days_in_week = 7
# t_max =  weeks * n_days_in_week # Maximum simulation time(weeks)
# dt = weeks/t_max # Time step

# # Forward Euler method
# timesteps = int(t_max / dt)
# time = np.linspace(0, t_max, timesteps)


