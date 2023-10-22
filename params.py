import numpy as np

def defined_params():
	k1 = 2.34 * 10**-5 # rho2 https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009839
	k2 = 234 * 10**-5 * day_conversion # day combi model
	k4 = 280 * 10**-5 * day_conversion # day combi model
	k7 = 50 #* 10**-5 # k1 and k2 between 30-60 https://zero.sci-hub.se/3771/b68709ea5f5640da4199e36ff25ef036/cumming2009.pdf
	k8 = 30 #* 10**-5 # k1 and k2 between 30-60 https://zero.sci-hub.se/3771/b68709ea5f5640da4199e36ff25ef036/cumming2009.pdf
	k11 = 2 * 10**(-7) * day_conversion # https://www.sciencedirect.com/science/article/pii/S0045782516302857?casa_token=ByHEzHgojSEAAAAA:XNdfPARqEPtiO3rcqb0jo9d--utWdu-swPxNKOLyK5huphzY4TcRxiVo4c4yzCASMY-tOswVpzY#br000425
	lambda1 = 0.001 * day_conversion # https://www.sciencedirect.com/science/article/pii/S0045782516302857?casa_token=ByHEzHgojSEAAAAA:XNdfPARqEPtiO3rcqb0jo9d--utWdu-swPxNKOLyK5huphzY4TcRxiVo4c4yzCASMY-tOswVpzY#br000435
	rho1 = 0.3 # https://link.springer.com/article/10.1007/s11538-012-9751-z/tables/1
	mu1 = 0.07 # day-1 mu_AM https://link.springer.com/article/10.1186/1471-2105-14-S6-S7/tables/2
	mu2 = 7 # day-1 mu_CH https://link.springer.com/article/10.1186/1471-2105-14-S6-S7/tables/2
	mu5 = 0.1 # https://link.springer.com/article/10.1007/s11538-012-9751-z/tables/1
	mu7 = 9.7 * 10**(-5) * day_conversion # https://www.sciencedirect.com/science/article/pii/S0045782516302857?casa_token=ByHEzHgojSEAAAAA:XNdfPARqEPtiO3rcqb0jo9d--utWdu-swPxNKOLyK5huphzY4TcRxiVo4c4yzCASMY-tOswVpzY#br000475
	mu8 = 9.7 * 10**(-5) * day_conversion # https://www.sciencedirect.com/science/article/pii/S0045782516302857?casa_token=ByHEzHgojSEAAAAA:XNdfPARqEPtiO3rcqb0jo9d--utWdu-swPxNKOLyK5huphzY4TcRxiVo4c4yzCASMY-tOswVpzY#br000475
	# Conversion parameters
	gamma1 = 10**(-5)
	gamma2 = 10**(-5)
	gamma3 = 10**(-5)
	gamma4 = 10**(-5)
	gamma5 = 10**(-5)
	gamma6 = 10**(-5)
	gamma7 = 10**(-5)
	gamma8 = 10**(-5)
	gamma9 = 10**(-5)
	zeta1 = -10**(-5)
	zeta2 = 10**(-5)
	zeta3 = 10**(-5)
	zeta4 = 10**(-5)
	zeta5 = 10**(-5)
	A_Malpha0 = 0
	CIII0 = 0
	CI0 = 0
	f_dillution = 1/16 #
	return k1, k2, k4, k7, k8, k11, lambda1, rho1, mu1, mu2, mu5, mu7, mu8, gamma1, gamma2, gamma3, gamma4, gamma5, gamma6, gamma7, gamma8, zeta1, zeta2, zeta3, zeta4, zeta5, f_dillution, A_Malpha0, CIII0, CI0

def initial_parameters():
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
	gamma1 = 10**(-5)
	gamma2 = 10**(-5)
	gamma3 = 10**(-5)
	gamma4 = 10**(-5)
	gamma5 = 10**(-5)
	gamma6 = 10**(-5)
	gamma7 = 10**(-5)
	gamma8 = 10**(-5)
	gamma9 = 10**(-5)
	zeta1 = 10**(-5)
	zeta2 = 10**(-5)
	zeta3 = 10**(-5)
	zeta4 = 10**(-5)
	zeta5 = 10**(-5)
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

	parameters = {
        'k1': k1, 'k2': k2, 'k3': k3, 'k4': k4, 'k5': k5, 'k6': k6, 'k7': k7, 'k8': k8, 'k9': k9, 'k10': k10, 'k11': k11,
        'gamma1': gamma1, 'gamma2': gamma2, 'gamma3': gamma3, 'gamma4': gamma4, 'gamma5': gamma5, 'gamma6': gamma6, 'gamma7': gamma7, 'gamma8': gamma8,
        'zeta1': zeta1, 'zeta2': zeta2, 'zeta3': zeta3, 'zeta4': zeta4, 'zeta5': zeta5, 'f_dillution': f_dillution,
        'lambda1': lambda1, 'lambda2': lambda2, 'lambda3': lambda3, 'lambda4': lambda4,
        'rho1': rho1, 'rho2': rho2, 'rho3': rho3,
        'mu1': mu1, 'mu2': mu2, 'mu3': mu3, 'mu4': mu4, 'mu5': mu5, 'mu6': mu6, 'mu7': mu7, 'mu8': mu8,
        'upsilon1': upsilon1, 'upsilon2': upsilon2, 'upsilon3': upsilon3, 'upsilon4': upsilon4,
        'omega1': omega1, 'omega2': omega2, 'omega3': omega3,
        'A_MII0': A_MII0, 'I0': I0, 'beta0': beta0, 'A_MC0': A_MC0, 'A_F0': A_F0, 'A_M0': A_M0, 'A_Malpha0': A_Malpha0,
        'CIII0': CIII0, 'CI0': CI0
    }
	return parameters

# Time parameters
weeks = 30
n_days_in_week = 7
t_max =  weeks * n_days_in_week # Maximum simulation time(weeks)
dt = weeks/t_max # Time step

# Forward Euler method
timesteps = int(t_max / dt)
time = np.linspace(0, t_max, timesteps)