from main import *




# interesting metrics 
# fibroblast activity
# collagen type I to type III ratio
# cell proliferation rate
# intensity of the inflammatory response
# wound contraction metric(production of myofibroblasts)
# celular senescence(loss of proliferative capacity over time)
# excessive fibroblast activity = fibrosis
# print(A_MII1)

# import matplotlib.pyplot as plt

# # Metrics Calculation
# fibroblast_activity = A_F1[1:]  # or A_F2 depending on the scenario
# collagen_ratio = [a_m / ciii for a_m, ciii in zip(A_M1[1:], CIII1[1:])]  # or A_M2, CIII2
# # Calculate proliferation rate
# proliferation_rate = np.diff(fibroblast_activity) / dt
# # Time points for proliferation rate (one less than the time array due to np.diff)
# proliferation_time = time[1:]
# inflammatory_response = I1[1:]  # or CI2
# wound_contraction = A_M1[1:]  # or A_MC2
# cellular_senescence = [1 - (a_m / A_M0) for a_m in A_M1[1:]]  # or A_M2, A_M0
# fibrosis = [a_f - A_F0 for a_f in A_F1[1:]]  # or A_F2, A_F0



# # Metrics Calculation
# fibroblast_activity2 = A_F2[1:]  # or A_F2 depending on the scenario
# collagen_ratio2 = [a_m / ciii for a_m, ciii in zip(A_M2[1:], CIII2[1:])]  # or A_M2, CIII2
# # Calculate proliferation rate
# proliferation_rate2 = np.diff(fibroblast_activity2) / dt
# # Time points for proliferation rate (one less than the time array due to np.diff)
# proliferation_time2 = time[1:]
# inflammatory_response2 = I2[1:]  # or CI2
# wound_contraction2 = A_M2[1:]  # or A_MC2
# cellular_senescence2 = [1 - (a_m / A_M0) for a_m in A_M2[1:]]  # or A_M2, A_M0
# fibrosis2 = [a_f - A_F0 for a_f in A_F2[1:]]  # or A_F2, A_F0



# # pro_anti_inflamatory_ratio = [i / t for i, t in zip(I2[1:], beta2[1:])]
# pro_anti_inflamatory_differential = np.diff(np.asarray(I2[1:]) - np.asarray(beta2[1:]))/dt

# # plt.plot(time[1:], pro_anti_inflamatory_differential)
# # plt.show()
# # # Calculate weeks
# # weeks = 30
# # days_in_week = 7
# # total_days = weeks * days_in_week
# # days_per_interval = 6 * days_in_week

# # # Calculate intervals
# # intervals = total_days // days_per_interval




# # # Split data into intervals
# # collagen_ratio_intervals = np.array_split(collagen_ratio, intervals)
# # cellular_senescence_intervals = np.array_split(cellular_senescence, intervals)
# # fibrosis_intervals = np.array_split(fibrosis, intervals)

# # # Calculate mean and standard deviation for each interval
# # mean_collagen_ratio = [np.mean(interval) for interval in collagen_ratio_intervals]
# # std_collagen_ratio = [np.std(interval) for interval in collagen_ratio_intervals]
# # mean_cellular_senescence = [np.mean(interval) for interval in cellular_senescence_intervals]
# # std_cellular_senescence = [np.std(interval) for interval in cellular_senescence_intervals]
# # mean_fibrosis = [np.mean(interval) for interval in fibrosis_intervals]
# # std_fibrosis = [np.std(interval) for interval in fibrosis_intervals]




# # # Visualization
# # plt.figure(figsize=(10, 8))

# # # Collagen Ratio Box Plot
# # plt.subplot(3, 1, 1)
# # plt.boxplot(collagen_ratio_intervals, showmeans=True)
# # plt.xticks(np.arange(1, intervals + 1), ['Week {}'.format(i) for i in range(1, intervals + 1)])
# # plt.xlabel('Intervals')
# # plt.ylabel('Collagen Ratio')
# # plt.title('Collagen Ratio Per 6 Weeks')
# # plt.grid(True)

# # # Cellular Senescence Box Plot
# # plt.subplot(3, 1, 2)
# # plt.boxplot(cellular_senescence_intervals, showmeans=True)
# # plt.xticks(np.arange(1, intervals + 1), ['Week {}'.format(i) for i in range(1, intervals + 1)])
# # plt.xlabel('Intervals')
# # plt.ylabel('Cellular Senescence')
# # plt.title('Cellular Senescence Per 6 Weeks')
# # plt.grid(True)

# # # Fibrosis Box Plot
# # plt.subplot(3, 1, 3)
# # plt.boxplot(fibrosis_intervals, showmeans=True)
# # plt.xticks(np.arange(1, intervals + 1), ['Week {}'.format(i) for i in range(1, intervals + 1)])
# # plt.xlabel('Intervals')
# # plt.ylabel('Fibrosis')
# # plt.title('Fibrosis Per 6 Weeks')
# # plt.grid(True)

# # plt.tight_layout()
# # plt.show()

# # Visualization
# plt.figure(figsize=(12, 8))

# plt.subplot(3, 2, 1)
# plt.plot(time, fibroblast_activity, label="scenario 1")
# plt.plot(time, fibroblast_activity2, label="scenario 1")
# plt.xlabel('Time')
# plt.ylabel('Fibroblast Activity')
# plt.title('Fibroblast Activity Over Time')

# plt.subplot(3, 2, 2)
# plt.plot(time, collagen_ratio, label="scenario 1")
# plt.plot(time, collagen_ratio2, label="scenario 2")
# plt.xlabel('Time')
# plt.ylabel('Collagen Type I to Type III Ratio')
# plt.title('Collagen Ratio Over Time')

# plt.subplot(3, 2, 3)
# plt.plot(proliferation_time, proliferation_rate, label="scenario 1")
# plt.plot(proliferation_time2, proliferation_rate2, label="scenario 2")
# plt.xlabel('Time')
# plt.ylabel('Cell Proliferation Rate')
# plt.title('Cell Proliferation Rate Over Time')

# plt.subplot(3, 2, 4)
# plt.plot(time, inflammatory_response, label="scenario 1")
# plt.plot(time, inflammatory_response2, label="scenario 2")
# plt.xlabel('Time')
# plt.ylabel('Inflammatory Response')
# plt.title('Inflammatory Response Over Time')

# plt.subplot(3, 2, 5)
# plt.plot(time, wound_contraction, label="scenario 1")
# plt.plot(time, wound_contraction2, label="scenario 2")
# plt.xlabel('Time')
# plt.ylabel('Wound Contraction Metric')
# plt.title('Wound Contraction Metric Over Time')

# plt.subplot(3, 2, 6)
# plt.plot(time, cellular_senescence, label="scenario 1")
# plt.plot(time, cellular_senescence2, label="scenario 2")
# plt.xlabel('Time')
# plt.ylabel('Cellular Senescence')
# plt.title('Cellular Senescence Over Time')
# plt.legend(loc = "best")
# plt.tight_layout()
# plt.show()
