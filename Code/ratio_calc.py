import numpy as np
import matplotlib.pyplot as plt
from main import * 
from params import initial_parameters
from main import solver
import pickle


def get_params(df, params_wanted=None):
    AF0params = list()
    AM0params = list()
    AMalpha0params = list()
    AMC0params = list()
    AMII0params = list()
    beta0params = list()
    CI0params = list()
    CIII0params = list()
    I0params = list()

    keys_to_change = ["A_F0", "A_M0", "A_Malpha0", "A_MC0", "A_MII0", "beta0", "CI0", "CIII0", "I0"]

    if df == "max":
        with open("max_vals_df.pkl", "rb") as f:
            df = pickle.load(f)
            df.reset_index()
    else:
        with open("min_vals_df.pkl", "rb") as f:
            df = pickle.load(f)
            
    for key in keys_to_change:
        if key == "A_F0":
            AF0params.append(df[key][0::].values)
            if params_wanted == "A_F0":
                return AF0params
        if key == "I0":
            I0params.append(df[key][0::].values)
            if params_wanted == "I0":
                return I0params
        if key == "A_M0":
            AM0params.append(df[key][0::].values)
            if params_wanted == "A_M0":
                return AM0params
        if key == "A_Malpha0":
            AMalpha0params.append(df[key][0::].values)
            if params_wanted == "A_Malpha0":
                return AMalpha0params
        if key == "A_MC0":
            AMC0params.append(df[key][0::].values)
            if params_wanted == "A_MC0":
                return AMC0params
        if key == "A_MII0":
            AMII0params.append(df[key][0::].values)
            if params_wanted == "A_MII0":
                return AMII0params
        if key == "beta0":
            beta0params.append(df[key][0::].values)
            if params_wanted == "beta0":
                return beta0params
        if key == "CI0":
            CI0params.append(df[key][0::].values)
            if params_wanted == "CI0":
                return CI0params
        if key == "CIII0":
            CIII0params.append(df[key][0::].values)
            if params_wanted == "CIII0":
                return CIII0params
        
    return AF0params, AM0params, AMalpha0params, AMC0params, AMII0params, beta0params, CI0params, CIII0params, I0params

    
        

def replace_paramset(row, old_param_dict, df):
    '''
    row represents which row we want from the actual df
    '''
    keys_to_change = ["A_F0", "A_M0", "A_Malpha0", "A_MC0", "A_MII0", "beta0", "CI0", "CIII0", "I0"]
    # print("old params", [old_param_dict[key] for key in keys_to_change])
    for key in keys_to_change:
        old_param_dict[key] = get_params(df, params_wanted=key)[0][row]
    # print("new params", [old_param_dict[key] for key in keys_to_change])
    new_param_dict = old_param_dict
    return new_param_dict
# init_params = initial_parameters()
# replace_paramset(0, init_params)


def fig_1():
    init_params = initial_parameters()
    time, dt, A_MII1, I1, beta1, A_MC1, A_F1, A_M1, A_Malpha1, CIII1, CI1, A_MII2, I2, beta2, A_MC2, A_F2, A_M2, A_Malpha2, CIII2, CI2 = solver(init_params)

    solver_results = [
        (time, dt, A_MII1, I1, beta1, A_MC1, A_F1, A_M1, A_Malpha1, CIII1, CI1, A_MII2, I2, beta2, A_MC2, A_F2, A_M2, A_Malpha2, CIII2, CI2),
    ]

    # Prepare lists to store the ratios for plotting
    ratio_AFAM1 = []  # Ratio of A_F / A_M
    ratio_CIIICI1 = []  # Ratio of CIII / CI
    ratio_AFAM2 = []  # Ratio of A_F / A_M
    ratio_CIIICI2 = []  # Ratio of CIII / CI

    # Loop through each solver result to calculate the ratios
    for result in solver_results:
        # Unpack the relevant parameters
        A_F1 = result[6]
        A_M1 = result[7]
        CIII1 = result[9]
        CI1 = result[10]

        A_F2 = result[15]
        A_M2 = result[16]
        CIII2 = result[-2]
        CI2 = result[-1]
        
        
        # Calculate the ratios
        if A_M1 and A_M2 != 0:
            ratio_AFM1 = np.divide(A_F1, A_M1)
            ratio_CICIII1 = np.divide(CI1, CIII1)
            ratio_AFM2 = np.divide(A_F2, A_M2)
            ratio_CICIII2 = np.divide(CI2, CIII2)
            
            # Append the ratios to the lists
            ratio_AFAM1.append(ratio_AFM1)
            ratio_CIIICI1.append(ratio_CICIII1)
            ratio_AFAM2.append(ratio_AFM2)
            ratio_CIIICI2.append(ratio_CICIII2)



    # Generate x-values for the diagonal line
    x_values = np.linspace(1, max(ratio_AFM1), len(ratio_AFM1))  # Generate x-values from 0 to 5

    # Generate y-values for the diagonal line with noise
    # Start with a straight line from y=5 to y=0 (x=5)
    y_values = 6 - x_values

    # Add some periodical noise to y-values
    noise_amplitude = 0.3  # Amplitude of the noise
    frequency = 3  # Frequency of the noise (adjust for desired period)
    noise = noise_amplitude * np.sin(frequency * x_values)  # Periodical sine wave noise

    # Add noise to y-values
    y_values += noise

    # Plotting the ratios
    plt.figure(figsize=(8, 6))
    plt.scatter(ratio_AFAM1, ratio_CIIICI1, color="lightblue", alpha=0.5, label="Sc. 1", linewidth=0.6)
    plt.scatter(ratio_AFAM2, ratio_CIIICI2, color="steelblue", alpha=0.5, label="Sc. 2", linewidth=0.6)
    plt.scatter(x_values, y_values, color='springgreen', label='Healthy standard', linewidth=0.6, alpha=0.5)
    plt.xlabel('Ratio of A_F / A_M')
    plt.ylabel('Ratio of CI / CIII')
    plt.legend()

    # Save and display the plot
    plt.title('Relationship between A_M/A_F and CI/CIII Ratios')
    plt.savefig("ratio_calc_1.png", dpi=300)
    plt.show()



def fig_2(df = None):
    import os

    # Create a directory to store pickle files
    output_folder = "metric_plots"
    os.makedirs(output_folder, exist_ok=True)
    init_params = initial_parameters()

    dfs = ["max", "min"]
    rows = np.arange(0, 18)
    colors = ["lightblue", "steelblue"]

    
    for row in rows:
        for df in dfs:
            # Prepare lists to store results for plotting
            ratio_AFAM1 = list()  # Ratio of A_F / A_M
            ratio_CIIICI1 = list()  # Ratio of CI / CIII
            ratio_AFAM2 = list()  # Ratio of A_F / A_M
            ratio_CIIICI2 = list()  # Ratio of CI / CIII
            # Replace parameters in initial_params dictionary using replace_paramset function
            updated_params = replace_paramset(row, init_params, df)
            
            # Solve the system with updated parameters to get results
            time, dt, A_MII1, I1, beta1, A_MC1, A_F1, A_M1, A_Malpha1, CIII1, CI1, A_MII2, I2, beta2, A_MC2, A_F2, A_M2, A_Malpha2, CIII2, CI2 = solver(updated_params)
            
            # Calculate ratios (avoid division by zero)
            if A_F1 != 0 or CI1!=0:
                ratio_AFAM = np.divide(A_F1, A_M1)
                ratio_CIIICI = np.divide(CI1, CIII1)
                ratio_AFAM1.append(ratio_AFAM)
                ratio_CIIICI1.append(ratio_CIIICI)
            
            if A_F2 != 0 or CI1!=0:
                ratio_AFAM = np.divide(A_F2, A_M2)
                ratio_CIIICI = np.divide(CI2, CIII2)
                ratio_AFAM2.append(ratio_AFAM)
                ratio_CIIICI2.append(ratio_CIIICI)

            

            # Plotting the ratios
            plt.figure(figsize=(8, 6))
            plt.scatter(ratio_AFAM1, ratio_CIIICI1, color=colors[0], alpha=0.5, label=f"(Sc.1)", s = 15, marker = "o")
            plt.scatter(ratio_AFAM2, ratio_CIIICI2, color=colors[1], alpha=0.5, label=f"(Sc.2)", s = 10, marker = "o")
            plt.xlabel('Contraction ratio')
            plt.ylabel('Collagen ratio')
            plt.legend()
            # plt.title('Relationship between A_M/A_F and CI/CIII Ratios across Parameter Sets')
            plt.savefig(f"{output_folder}/{df}_ratiometrics_{row}.png", dpi=300)
            # plt.show()
        
def fig_3(df = None):
    import os

    # Create a directory to store pickle files
    output_folder = "metric_plots_disease"
    os.makedirs(output_folder, exist_ok=True)
    init_params = initial_parameters()

    dfs = ["max", "min"]
    rows_to_plot = []
    colors = ["lightblue", "steelblue"]

    
    for row in rows_to_plot:
        for df in dfs:
            # Prepare lists to store results for plotting
            ratio_AFAM1 = list()  # Ratio of A_F / A_M
            ratio_CIIICI1 = list()  # Ratio of CI / CIII
            ratio_AFAM2 = list()  # Ratio of A_F / A_M
            ratio_CIIICI2 = list()  # Ratio of CI / CIII
            # Replace parameters in initial_params dictionary using replace_paramset function
            updated_params = replace_paramset(row, init_params, df)
            
            # Solve the system with updated parameters to get results
            time, dt, A_MII1, I1, beta1, A_MC1, A_F1, A_M1, A_Malpha1, CIII1, CI1, A_MII2, I2, beta2, A_MC2, A_F2, A_M2, A_Malpha2, CIII2, CI2 = solver(updated_params)
            
            # Calculate ratios (avoid division by zero)
            if A_F1 != 0 or CI1!=0:
                ratio_AFAM = np.divide(A_F1, A_M1)
                ratio_CIIICI = np.divide(CI1, CIII1)
                ratio_AFAM1.append(ratio_AFAM)
                ratio_CIIICI1.append(ratio_CIIICI)
            
            if A_F2 != 0 or CI1!=0:
                ratio_AFAM = np.divide(A_F2, A_M2)
                ratio_CIIICI = np.divide(CI2, CIII2)
                ratio_AFAM2.append(ratio_AFAM)
                ratio_CIIICI2.append(ratio_CIIICI)

            

            # Plotting the ratios
            plt.figure(figsize=(8, 6))
            plt.scatter(ratio_AFAM1, ratio_CIIICI1, color=colors[0], alpha=0.5, label=f"(Sc.1)", s = 15, marker = "o")
            plt.scatter(ratio_AFAM2, ratio_CIIICI2, color=colors[1], alpha=0.5, label=f"(Sc.2)", s = 10, marker = "o")
            plt.xlabel('Contraction ratio')
            plt.ylabel('Collagen ratio')
            plt.legend()
            # plt.title('Relationship between A_M/A_F and CI/CIII Ratios across Parameter Sets')
            plt.savefig(f"{output_folder}/{df}_ratiometrics_{row}.png", dpi=300)
            # plt.show()
        
fig_3()