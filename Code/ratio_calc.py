import numpy as np
import matplotlib.pyplot as plt
from main import * 
from params import initial_parameters
from main import solver
import pickle
import os
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

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

def fig_1a():

    init_params = initial_parameters()
    ratio_AFAM1 = []  # Ratio of A_F / A_M for Sc.1
    ratio_CIIICI1 = []  # Ratio of CI / CIII for Sc.1
    ratio_AFAM2 = []  # Ratio of A_F / A_M for Sc.2
    ratio_CIIICI2 = []  # Ratio of CI / CIII for Sc.2
    
    
    time, dt, A_MII1, I1, beta1, A_MC1, A_F1, A_M1, A_Malpha1, CIII1, CI1, A_MII2, I2, beta2, A_MC2, A_F2, A_M2, A_Malpha2, CIII2, CI2 = solver(init_params)

    if A_F1 != 0 or CI1 != 0:
        ratio_AFAM1 = np.divide(A_F1, A_M1)
        ratio_CIIICI1 = np.divide(CI1, CIII1)
        ratio_PROINF1 = np.divide(I1, A_MC1, out=np.zeros_like(I1), where=A_MC1 != 0 or I1 != 0)
        ratio_ANTIINF1 = np.divide(beta1, A_MII1, out=np.zeros_like(beta1), where=A_MII1 != 0 or beta1 != 0)
        inflammatory_metric1 = ratio_PROINF1 - ratio_ANTIINF1

    if A_F2 != 0 or CI2 != 0:
        ratio_AFAM2 = np.divide(A_F2, A_M2)
        ratio_CIIICI2 = np.divide(CI2, CIII2)
        ratio_PROINF2 = np.divide(I2, A_MC2, out=np.zeros_like(I2), where=A_MC2 != 0 or I2 != 0)
        ratio_ANTIINF2 = np.divide(beta2, A_MII2, out=np.zeros_like(beta2), where=A_MII2 != 0 or beta2 != 0)
        inflammatory_metric2 = ratio_PROINF2 - ratio_ANTIINF2

    plt.figure(figsize=(8, 6))
    indices = np.arange(0, len(ratio_AFAM1), 4000)
    
    
    # Use 'summer' colormap based on inflammatory_metric1 values for Sc.1
    if len(ratio_AFAM1) > 0:
        plt.scatter(ratio_AFAM1[indices], ratio_CIIICI1[indices], c=inflammatory_metric1[indices], cmap='summer', alpha=1, label='Sc.1', marker='o')
    # Use 'summer' colormap based on inflammatory_metric2 values for Sc.2
    if len(ratio_AFAM2) > 0:
        plt.scatter(ratio_AFAM2[indices], ratio_CIIICI2[indices], c=inflammatory_metric2[indices], cmap='summer', alpha=1, label='Sc.2', marker='*')
    plt.scatter(ratio_AFAM1[-1], ratio_CIIICI1[-1], alpha=1, label='$t_{end}$, Sc.1', marker='o', color = "red")
    plt.scatter(ratio_AFAM2[-1], ratio_CIIICI2[-1], alpha=1, label='$t_{end}$, Sc.2', marker='*', color = "red")
            
    plt.xlabel(r'Contraction ratio ($\frac{A_F}{A_M}$)')
    plt.ylabel(r'Collagen ratio ($\frac{CI}{CIII}$)')
    plt.colorbar(label=r'Inflammation metric($\frac{I}{A_{MC}}-\frac{T}{A_{MII}}$)') 
    plt.legend()
    plt.ylim([0, 8])

    # Save and display the plot
    plt.savefig("ratio_calc_1.png", dpi=300)
    plt.show()



def fig_1b():

    # Create a directory to store pickle files
    output_folder = "metric_plots_inflammation"
    os.makedirs(output_folder, exist_ok=True)
    init_params = initial_parameters()  
    # Solve the system with updated parameters to get results
    time, dt, A_MII1, I1, beta1, A_MC1, A_F1, A_M1, A_Malpha1, CIII1, CI1, A_MII2, I2, beta2, A_MC2, A_F2, A_M2, A_Malpha2, CIII2, CI2 = solver(init_params)

    # Determine colors based on which line is higher
    plt.figure(figsize=(8, 6))
    
    if A_F1 != 0 or CI1 != 0:
        ratio_PROINF1 = np.divide(I1, A_MC1, out=np.zeros_like(I1), where=A_MC1 != 0 or I1 != 0)
        ratio_ANTIINF1 = np.divide(beta1, A_MII1, out=np.zeros_like(beta1), where=A_MII1 != 0 or beta1 != 0)

    if A_F2 != 0 or CI2 != 0:
        ratio_PROINF2 = np.divide(I2, A_MC2, out=np.zeros_like(I2), where=A_MC2 != 0 or I2 != 0)
        ratio_ANTIINF2 = np.divide(beta2, A_MII2, out=np.zeros_like(beta2), where=A_MII2 != 0 or beta2 != 0)

    # Normalize ratios to 1 relative to the maximum value of PROINF or ANTIINF
    max_value1 = max(np.max(ratio_PROINF1), np.max(ratio_ANTIINF1))
    ratio_PROINF1 /= max_value1
    ratio_ANTIINF1 /= max_value1
    # Normalize ratios to 1 relative to the maximum value of PROINF or ANTIINF
    max_value2 = max(np.max(ratio_PROINF2), np.max(ratio_ANTIINF2))
    ratio_PROINF2 /= max_value2
    ratio_ANTIINF2 /= max_value2
    # Plotting the ratios
    plt.plot(time, ratio_PROINF1[1::], color="red", alpha=0.5, label=f"Sc.1 - Pro-Inflammatory", linewidth=4)
    plt.plot(time, ratio_ANTIINF1[1::], color="blue", alpha=0.5, label=f"Sc.1 - Anti-Inflammatory",linewidth=4)
    # Plotting the ratios
    plt.plot(time, ratio_PROINF2[1::], color="red", alpha=0.5, label=f"Sc.2 - Pro-Inflammatory", linewidth=4, linestyle="dotted" )
    plt.plot(time, ratio_ANTIINF2[1::], color="blue", alpha=0.5, label=f"Sc.2 - Anti-Inflammatory", linewidth=4, linestyle="dotted")
    plt.xlabel('Time of simulation (days)')
    plt.ylabel(r'Inflammation metric($\frac{I}{A_{MC}}-\frac{T}{A_{MII}}$)')
    # plt.legend()
    plt.savefig(f"{output_folder}/init_ratiometrics_inf.png", dpi=300)
    plt.close()  # Close the plot to avoid displaying multiple plots in the loop

def fig_2(df=None):
    # Create a directory to store output files
    output_folder = "metric_plots"
    os.makedirs(output_folder, exist_ok=True)
    
    init_params = initial_parameters()

    dfs = ["max", "min"]
    rows = np.arange(0, 18, 1)

    for row in rows:
        for df_type in dfs:
            ratio_AFAM1 = []  # Ratio of A_F / A_M for Sc.1
            ratio_CIIICI1 = []  # Ratio of CI / CIII for Sc.1
            ratio_AFAM2 = []  # Ratio of A_F / A_M for Sc.2
            ratio_CIIICI2 = []  # Ratio of CI / CIII for Sc.2
            
            updated_params = replace_paramset(row, init_params, df_type)
            time, dt, A_MII1, I1, beta1, A_MC1, A_F1, A_M1, A_Malpha1, CIII1, CI1, A_MII2, I2, beta2, A_MC2, A_F2, A_M2, A_Malpha2, CIII2, CI2 = solver(updated_params)

            if A_F1 != 0 or CI1 != 0:
                ratio_AFAM1 = np.divide(A_F1, A_M1)
                ratio_CIIICI1 = np.divide(CI1, CIII1)
                ratio_PROINF1 = np.divide(I1, A_MC1, out=np.zeros_like(I1), where=A_MC1 != 0 or I1 != 0)
                ratio_ANTIINF1 = np.divide(beta1, A_MII1, out=np.zeros_like(beta1), where=A_MII1 != 0 or beta1 != 0)
                inflammatory_metric1 = ratio_PROINF1 - ratio_ANTIINF1

            if A_F2 != 0 or CI2 != 0:
                ratio_AFAM2 = np.divide(A_F2, A_M2)
                ratio_CIIICI2 = np.divide(CI2, CIII2)
                ratio_PROINF2 = np.divide(I2, A_MC2, out=np.zeros_like(I2), where=A_MC2 != 0 or I2 != 0)
                ratio_ANTIINF2 = np.divide(beta2, A_MII2, out=np.zeros_like(beta2), where=A_MII2 != 0 or beta2 != 0)
                inflammatory_metric2 = ratio_PROINF2 - ratio_ANTIINF2

            plt.figure(figsize=(8, 6))
            indices = np.arange(0, len(ratio_AFAM1), 4000)
            
            
            # Use 'summer' colormap based on inflammatory_metric1 values for Sc.1
            if len(ratio_AFAM1) > 0:
                # norm1 = Normalize(vmin=np.min(inflammatory_metric1), vmax=np.max(inflammatory_metric1))
                plt.scatter(ratio_AFAM1[indices], ratio_CIIICI1[indices], c=inflammatory_metric1[indices], cmap='summer', alpha=1, label='Sc.1', marker='o')
            # Use 'summer' colormap based on inflammatory_metric2 values for Sc.2
            if len(ratio_AFAM2) > 0:
                # norm2 = Normalize(vmin=np.min(inflammatory_metric2), vmax=np.max(inflammatory_metric2))
                plt.scatter(ratio_AFAM2[indices], ratio_CIIICI2[indices], c=inflammatory_metric2[indices], cmap='summer', alpha=1, label='Sc.2', marker='*')
            
            plt.scatter(ratio_AFAM1[-1], ratio_CIIICI1[-1], alpha=1, label='$t_{end}$, Sc.1', marker='o', color = "red")
            plt.scatter(ratio_AFAM2[-1], ratio_CIIICI2[-1], alpha=1, label='$t_{end}$, Sc.2', marker='*', color = "red")
                 
            plt.xlabel(r'Contraction ratio ($\frac{A_F}{A_M}$)')
            plt.ylabel(r'Collagen ratio ($\frac{CI}{CIII}$)')
            plt.colorbar(label=r'Inflammation metric($\frac{I}{A_{MC}}-\frac{T}{A_{MII}}$)') 
            plt.legend()
            # Save the plot to a file
            plt.savefig(f"{output_folder}/{df_type}_ratiometrics_{row}.png", dpi=300)
            plt.close()  # Close the figure to release memory
        
def fig_3(df=None):
    # Create a directory to store pickle files
    output_folder = "metric_plots_inflammation"
    os.makedirs(output_folder, exist_ok=True)
    init_params = initial_parameters()  # Assuming you have defined this function

    dfs = ["max", "min"]
    rows = [3, 13, 15]#np.arange(0, 18) #[14, 15, 16, 17]
    for row in rows:
        for df_type in dfs:
            # Replace parameters in initial_params dictionary using replace_paramset function
            updated_params = replace_paramset(row, init_params, df_type)
            
            # Solve the system with updated parameters to get results
            time, dt, A_MII1, I1, beta1, A_MC1, A_F1, A_M1, A_Malpha1, CIII1, CI1, A_MII2, I2, beta2, A_MC2, A_F2, A_M2, A_Malpha2, CIII2, CI2 = solver(updated_params)
        
            # Determine colors based on which line is higher
            plt.figure(figsize=(8, 6))
            
            if A_F1 != 0 or CI1 != 0:
                ratio_PROINF1 = np.divide(I1, A_MC1, out=np.zeros_like(I1), where=A_MC1 != 0 or I1 != 0)
                ratio_ANTIINF1 = np.divide(beta1, A_MII1, out=np.zeros_like(beta1), where=A_MII1 != 0 or beta1 != 0)

            if A_F2 != 0 or CI2 != 0:
                ratio_PROINF2 = np.divide(I2, A_MC2, out=np.zeros_like(I2), where=A_MC2 != 0 or I2 != 0)
                ratio_ANTIINF2 = np.divide(beta2, A_MII2, out=np.zeros_like(beta2), where=A_MII2 != 0 or beta2 != 0)
            
            # Normalize ratios to 1 relative to the maximum value of PROINF or ANTIINF
            max_value1 = max(np.max(ratio_PROINF1), np.max(ratio_ANTIINF1))
            ratio_PROINF1 /= max_value1
            ratio_ANTIINF1 /= max_value1
             # Normalize ratios to 1 relative to the maximum value of PROINF or ANTIINF
            max_value2 = max(np.max(ratio_PROINF2), np.max(ratio_ANTIINF2))
            ratio_PROINF2 /= max_value2
            ratio_ANTIINF2 /= max_value2
            # Plotting the ratios
            plt.plot(time, ratio_PROINF1[1::], color="red", alpha=0.5, label=f"Sc.1 - Pro-Inflammatory", linewidth=4)
            plt.plot(time, ratio_ANTIINF1[1::], color="blue", alpha=0.5, label=f"Sc.1 - Anti-Inflammatory",linewidth=4)
            # Plotting the ratios
            plt.plot(time, ratio_PROINF2[1::], color="red", alpha=0.5, label=f"Sc.2 - Pro-Inflammatory", linewidth=4, linestyle="dotted" )
            plt.plot(time, ratio_ANTIINF2[1::], color="blue", alpha=0.5, label=f"Sc.2 - Anti-Inflammatory", linewidth=4, linestyle="dotted")
            plt.xlabel('Time of simulation (days)')
            plt.ylabel(r'Normalized concentration(AU)')
            plt.legend()
            plt.savefig(f"{output_folder}/{df_type}_ratiometrics_inf_{row}.png", dpi=300)
            plt.close()  # Close the plot to avoid displaying multiple plots in the loop



def pca_data():
    # Create a directory to store PCA plots
    output_folder = "PCA"
    os.makedirs(output_folder, exist_ok=True)
    

    # Define empty lists to store final ratios and inflammatory metrics for each scenario
    ratios_sc1 = []
    ratios_sc2 = []
    inflammatory_metrics_sc1 = []
    inflammatory_metrics_sc2 = []
    updated_params = initial_parameters()

    for i in range(999):
        # Load results from the specified experiment (
        with open(f"pickle_plasma3/experiment_{i}_results.pkl", "rb") as f:
            loaded_results = pickle.load(f)
        
        # Extract parameter iterations for scenario 1 and scenario 2
        parameters_scenario1 = loaded_results["initial_values1"]
        parameters_scenario2 = loaded_results["initial_values2"]
        
        

        # Iterate over parameter iterations for Scenario 1
        updated_params.update(parameters_scenario1)
        # Solve the system with updated parameters to get results
        time, dt, A_MII1, I1, beta1, A_MC1, A_F1, A_M1, A_Malpha1, CIII1, CI1, A_MII2, I2, beta2, A_MC2, A_F2, A_M2, A_Malpha2, CIII2, CI2 = solver(updated_params)
        
        # Calculate ratios and metrics for Scenario 1
        if I1 != 0 or beta1 != 0:
            ratio_AFAM1 = np.divide(A_F1, A_M1)
            ratio_CIIICI1 = np.divide(CI1, CIII1)
            ratio_PROINF1 = np.divide(I1, A_MC1, out=np.zeros_like(I1), where=A_MC1 != 0 or I1 != 0)
            ratio_ANTIINF1 = np.divide(beta1, A_MII1, out=np.zeros_like(beta1), where=A_MII1 != 0 or beta1 != 0)
            inflammatory_metric1 = ratio_PROINF1 - ratio_ANTIINF1
            ratios_sc1.append([ratio_AFAM1[-1], ratio_CIIICI1[-1]])
            inflammatory_metrics_sc1.append(inflammatory_metric1[-1])

        
        # Replace parameters in initial_params dictionary
        updated_params.update(parameters_scenario2)
        
        # Solve the system with updated parameters to get results
        time, dt, A_MII1, I1, beta1, A_MC1, A_F1, A_M1, A_Malpha1, CIII1, CI1, A_MII2, I2, beta2, A_MC2, A_F2, A_M2, A_Malpha2, CIII2, CI2 = solver(updated_params)
        
        # Calculate ratios and metrics for Scenario 2
        if I2 != 0 or beta2 != 0:
            ratio_AFAM2 = np.divide(A_F2, A_M2)
            ratio_CIIICI2 = np.divide(CI2, CIII2)
            ratio_PROINF2 = np.divide(I2, A_MC2, out=np.zeros_like(I2), where=A_MC2 != 0 or I2 != 0)
            ratio_ANTIINF2 = np.divide(beta2, A_MII2, out=np.zeros_like(beta2), where=A_MII2 != 0 or beta2 != 0)
            inflammatory_metric2 = ratio_PROINF2 - ratio_ANTIINF2
            ratios_sc2.append([ratio_AFAM2[-1], ratio_CIIICI2[-1]])
            inflammatory_metrics_sc2.append(inflammatory_metric2[-1])
    
    # Convert lists to numpy arrays
    ratios_sc1 = np.array(ratios_sc1)
    with open(f"{output_folder}/sc1_rawratios_pca.pkl", "wb") as f:
        pickle.dump(ratios_sc1, f)
    ratios_sc2 = np.array(ratios_sc2)
    with open(f"{output_folder}/sc2_rawratios_pca.pkl", "wb") as f:
        pickle.dump(ratios_sc2, f)
    inflammatory_metrics_sc1 = np.array(inflammatory_metrics_sc1)
    with open(f"{output_folder}/sc1_rawinfratios_pca.pkl", "wb") as f:
        pickle.dump(inflammatory_metrics_sc1, f)
    inflammatory_metrics_sc2 = np.array(inflammatory_metrics_sc2)
    with open(f"{output_folder}/sc2_rawinfratios_pca.pkl", "wb") as f:
        pickle.dump(inflammatory_metrics_sc2, f)


    # Handle NaN values by replacing them with zeros
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    ratios_sc1 = imputer.fit_transform(ratios_sc1)
    ratios_sc2 = imputer.fit_transform(ratios_sc2)
    
    # Scale the ratios for better visualization
    scaler = MinMaxScaler()
    ratios_sc1_scaled = scaler.fit_transform(ratios_sc1)
    with open(f"{output_folder}/sc1_scaled_ratios_pca.pkl", "wb") as f:
        pickle.dump(ratios_sc1_scaled, f)
    ratios_sc2_scaled = scaler.fit_transform(ratios_sc2)
    with open(f"{output_folder}/sc2_scaled_ratios_pca.pkl", "wb") as f:
        pickle.dump(ratios_sc2_scaled, f)



def fig_4():
    output_folder = "PCA"
    scenarios = [1, 2]

    for scenario in scenarios:
        with open(f"{output_folder}/sc{scenario}_rawratios_pca.pkl", "rb") as f:
            ratios_sc_scaled = pickle.load(f)

        with open(f"{output_folder}/sc{scenario}_rawinfratios_pca.pkl", "rb") as f:
            inflammatory_metrics_sc = pickle.load(f) 

        # Handle NaN values by replacing them with zeros
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        transformed_data_sc = imputer.fit_transform(ratios_sc_scaled)
        inflammatory_metrics_sc = inflammatory_metrics_sc.reshape(-1, 1)
        inflammatory_metrics_sc = imputer.fit_transform(inflammatory_metrics_sc)

        # Define a function to assign labels based on data ranges
        def assign_label(x, y):
            if 1 < x < 9 and 5.5 < y < 6.2:
                return 'Normal'
            elif 5.5 < y < 6.5 and 0.8 < x < 12:
                return 'Normotrophic'
            elif 0 < x <= 0.8 and 5.5 < y < 12:
                return 'Hypertrophic'
            elif 1 < x < 12 and 0 < y < 4:
                return 'Keloid'
            else:
                return 'Other'  # Assign 'Other' label to data points outside defined ranges

        # Apply label assignment function to each data point
        labels = np.array([assign_label(x, y) for x, y in ratios_sc_scaled])

        # Define color mapping for labels
        label_color_map = {
            'Normal': {'alpha': 1, 'marker': '*'},
            'Normotrophic': {'alpha': 1, 'marker': '^'},
            'Hypertrophic': {'alpha': 1, 'marker': 'o'},
            'Keloid': {'alpha': 1, 'marker': 'D'},
            'Other': {'alpha': 1, 'marker': '.'}
        }

        # Scale the ratios for better visualization
        scaler = MinMaxScaler()
        transformed_data_sc = scaler.fit_transform(transformed_data_sc)
        inflammatory_metrics_sc = scaler.fit_transform(inflammatory_metrics_sc)

        # Apply t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)  # Set random_state for reproducibility
        transformed_data_sc = tsne.fit_transform(transformed_data_sc)

        # Plot t-SNE results colored by assigned labels
        plt.figure(figsize=(8, 6))

        for label, props in label_color_map.items():
            plt.scatter(transformed_data_sc[labels == label, 0], 
                        transformed_data_sc[labels == label, 1], 
                        label=label, alpha=props['alpha'], marker=props['marker'], 
                        c=inflammatory_metrics_sc[labels == label], cmap="inferno")

        plt.xlabel('Contraction Ratio Component')
        plt.ylabel('Collagen Ratio Component')
        plt.colorbar(label=r'Inflammation metric($\frac{I}{A_{MC}}-\frac{T}{A_{MII}}$)') 
        plt.legend(loc="upper left")

        # Manually adjust legend handles to display all icons in black
        legend = plt.legend()
        for handle in legend.legendHandles:
            handle.set_color('black')

        # plt.title(f't-SNE Analysis - Scenario {scenario}')
        plt.savefig(f"{output_folder}/tsne_plot_scenario_{scenario}_custom_labels.png", dpi=300)
        plt.show()
fig_1a()