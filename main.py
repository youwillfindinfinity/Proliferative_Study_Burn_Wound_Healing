import numpy as np
import matplotlib.pyplot as plt

# Production Parameters 
k1 = 0.1
k2 = 0.2
k3 = 0.15
k4 = 0.25
k5 = 0.2
k6 = 0.3
k7 = 0.1
k8 = 0.01
k9 = 0.23
k10 = 0.1
k11 = 0.33

# Conversion parameters
gamma1 = 0.05
gamma2 = 0.06
gamma3 = 0.04
gamma4 = 0.03
gamma5 = 0.02
gamma6 = 0.01
gamma8 = 0.06
zeta1 = 0.02
zeta2 = 0.03
zeta3 = 0.01
zeta4 = 0.04
zeta5 = 0.02


# Activation parameters
lambda1 = 0.06
lambda2 = 0.07
lambda3 = 0.08
lambda4 = 0.03

# Transition parameters
rho1 = 0.05
rho2 = 0.02
rho3 = 0.01

# Decay parameters
mu1 = 0.01
mu2 = 0.05
mu3 = 0.03
mu4 = 0.01
mu5 = 0.04
mu6 = 0.04
mu6 = 0.03
mu7 = 0.01
mu8 = 0.015

# sinusoidal parameters
upsilon1 = 0.02
upsilon2 = 0.03
upsilon3 = 0.01
upsilon4 = 0.02
omega1 = 0.5
omega2 = 0.7
omega3 = 0.6

# Initial conditions
A_MII0 = 1000
I0 = 10**(-7)
beta0 = 10**(-8)
A_MC0 = 100
A_F0 = 100
A_M0 = 20
A_Malpha0 = 0
CIII0 = 0
CI0 = 0

# Time parameters
t_max = 7  # Maximum simulation time(weeks)
n_days_in_week = 7
dt = t_max/n_days_in_week   # Time step

# Forward Euler method
timesteps = int(t_max / dt)
time = np.linspace(0, t_max, timesteps)

# Scenario 1 equations
def scenario1_equations(A_MII, I, beta, A_MC, A_F, A_M, A_Malpha, CIII, CI, t):
    A_MII_next = A_MII + k1 * np.exp(-mu1 * t)
    I_next = I + dt * (-k2 * upsilon1 * np.exp(-upsilon1 * t) + k6 * gamma1 * A_MC - mu2 * I)
    beta_next = beta + dt * (k3 * upsilon2 * np.exp(upsilon2 * t) + k4 * gamma2 * A_MII + k5 * gamma3 * A_MC - mu3 * beta)
    A_MC_next = A_MC + dt * (A_MC * I * lambda3 * zeta1 - mu7 * A_MC)
    A_F_next = A_F + dt * (lambda2 * zeta2 * I * A_F - lambda1 * zeta3 * beta * A_F - rho1 * A_F - mu5 * A_F)
    A_M_next = A_M + dt * (rho1 * A_F + lambda1 * zeta4 * beta * A_F + lambda4 * zeta5 * A_F * A_M - mu6 * A_M)
    A_Malpha_next = A_Malpha + dt * (rho2 * A_M)
    CIII_next = CIII + dt * (k7 * gamma1 * A_F + k10 * gamma1 * A_M - mu7 * CIII)
    CI_next = CI + dt * (k11 * gamma1 * A_Malpha - mu8 * CI)
    return A_MII_next, I_next, beta_next, A_MC_next, A_F_next, A_M_next, A_Malpha_next, CIII_next, CI_next

# Scenario 2 equations
def scenario2_equations(A_MII, I, beta, A_MC, A_F, A_M, A_Malpha, CIII, CI, t):
    A_MII_next = A_MII + dt * (k1 * np.exp(-mu1 * t) * np.cos(omega1 * t))
    I_next = I + dt * (-k2 * upsilon3 * np.exp(-upsilon3 * t) * np.cos(omega2 * t)
                       - k2 * np.exp(-upsilon3 * t) * omega2 * np.sin(omega2 * t) + k6 * gamma1 * A_MC - mu2 * I)
    beta_next = beta + dt * (k3 * np.exp(-upsilon4 * t) * omega3 * np.cos(omega3 * t)
                             - k3 * upsilon4 * np.exp(-upsilon4 * t) * np.sin(omega3 * t)
                             + k4 * gamma2 * A_MII + k5 * gamma3 * A_MC - mu3 * beta)
    A_MC_next = A_MC + dt * (A_MC * I * lambda3 * zeta1 - mu7 * A_MC)
    A_F_next = A_F + dt * (lambda2 * zeta2 * I * A_F - lambda1 * zeta3 * beta * A_F - rho1 * A_F - mu5 * A_F)
    A_M_next = A_M + dt * (rho1 * A_F + lambda1 * zeta4 * beta * A_F + lambda4 * zeta5 * A_F * A_M - mu6 * A_M)
    A_Malpha_next = A_Malpha + dt * (rho2 * A_M)
    CIII_next = CIII + dt * (k7 * gamma1 * A_F + k10 * gamma1 * A_M - mu7 * CIII)
    CI_next = CI + dt * (k11 * gamma1 * A_Malpha - mu8 * CI)
    return A_MII_next, I_next, beta_next, A_MC_next, A_F_next, A_M_next, A_Malpha_next, CIII_next, CI_next

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

# Plotting concentrations of cytokines, proteins, and cell counts for agents separately for each scenario
plt.figure(figsize=(15, 12))

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
plt.plot(time, CI1[1:], label=r'$C_I$(Scenario 1)')
plt.legend()
plt.title(r'$C_I$ concentration over time')
plt.xlabel('Time')
plt.ylabel('Concentration')

plt.subplot(3, 3, 4)
plt.plot(time, A_MII1[1:], label=r'$A_{MII}$(Scenario 1)')
plt.legend()
plt.title(r'$A_{MII}$ cell count over time')
plt.xlabel('Time')
plt.ylabel('Count')

plt.subplot(3, 3, 5)
plt.plot(time, A_MC1[1:], label=r'$A_{MC}$(Scenario 1)')
plt.legend()
plt.title(r'$A_{MC}$ cell count over time')
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
plt.title(r'$A_M\alpha$ cell count over time')
plt.xlabel('Time')
plt.ylabel('Count')

plt.subplot(3, 3, 8)
plt.plot(time, CIII1[1:], label=r'$C_{III}(Scenario 1)$')
plt.legend()
plt.title(r'$C_{III}$ concentration over time ')
plt.xlabel('Time')
plt.ylabel('Concentration')


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
plt.plot(time, CI2[1:], label=r'$C_{I}$(Scenario 2)')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Concentration')

plt.subplot(3, 3, 4)
plt.plot(time, A_MII2[1:], label=r'$A_{MII}$(Scenario 2)')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Count')

plt.subplot(3, 3, 5)
plt.plot(time, A_MC2[1:], label=r'$A_{MC}$(Scenario 2)')
plt.legend()
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

plt.tight_layout()
plt.show()
