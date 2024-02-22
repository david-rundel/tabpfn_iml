import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import pickle

experiment_results_path= 'experiments/kernel_shap/results/kernel_shap_900_20240222_012027.pkl'

with open(experiment_results_path, 'rb') as file:
    exp_results = pickle.load(file)

rectify_indices= True
if rectify_indices:
    loss_exact_marg_runs= exp_results["loss_exact_marginalization_per_run"]
    loss_appr_marg_runs= exp_results["loss_appr_marginalization_per_run"]
    max_s= exp_results["experiment_hyperparameters"]["max_s"]
    runs= exp_results["experiment_hyperparameters"]["runs"]


    appr_cols= list(range(1, max_s+1))
    exact_col= [-1]
    exp_results["loss_exact_marginalization_mean"]= pd.DataFrame(np.concatenate([loss_exact_marg_runs[i] for i in range(runs)], axis=1).mean(axis=-1), columns= exact_col, index= loss_exact_marg_runs[0].index)
    exp_results["loss_appr_marginalization_mean"]= pd.DataFrame(np.stack([loss_appr_marg_runs[i] for i in range(runs)]).mean(axis=0), columns= appr_cols, index= loss_appr_marg_runs[0].index)
    exp_results["loss_exact_marginalization_std"]= pd.DataFrame(np.concatenate([loss_exact_marg_runs[i] for i in range(runs)], axis=1).std(axis=-1), columns= exact_col, index= loss_exact_marg_runs[0].index)
    exp_results["loss_appr_marginalization_std"]= pd.DataFrame(np.stack([loss_appr_marg_runs[i] for i in range(runs)]).std(axis=0), columns= appr_cols, index= loss_appr_marg_runs[0].index)


def plot_results_M_L_Loss(loss_appr_marg,
                 loss_exact_marg):
    #TODO: Integrate weights

    plot_df= loss_appr_marg.copy()
    plot_df[-1]= loss_exact_marg.copy()

    x, y = np.meshgrid(plot_df.columns, plot_df.index)
    x_flat = x.flatten()
    y_flat = y.flatten()
    values = plot_df.values.flatten()

    plt.figure(figsize=(10, 6))

    cmap_original = plt.get_cmap('magma')
    cmap_inverted = ListedColormap(cmap_original.colors[::-1])

    scatter = plt.scatter(x_flat, y_flat, c=values, cmap=cmap_inverted, s=30)
    plt.colorbar(scatter, label='Values')
    plt.title('Loss per M and L')
    plt.xlabel('L')
    plt.ylabel('M')
    plt.show()

    del plot_df

# plot_each_run= False

# # #Plot overall results
# plot_results_M_L_Perf(exp_results["loss_appr_marginalization_mean"],
#              exp_results["loss_exact_marginalization_mean"])

# #Plot results per run
# if plot_each_run:
#     for run in range(runs):
#         plot_results_M_L_Perf(loss_appr_marg_runs[run],
#                      loss_exact_marg_runs[run])




n_train= exp_results["experiment_hyperparameters"]["n_train"]
n_inf= exp_results["experiment_hyperparameters"]["n_test"]

#Aggregate results of approximate marginalization
stacked_results_appr_mean= exp_results["loss_appr_marginalization_mean"].stack().reset_index()
stacked_results_appr_mean.columns = ['M', 'L', 'Loss_Mean']

stacked_results_appr_std= exp_results["loss_appr_marginalization_std"].stack().reset_index()
stacked_results_appr_std.columns = ['M', 'L', 'Loss_Std']

stacked_results_appr= pd.merge(stacked_results_appr_mean, stacked_results_appr_std, on=['M', 'L'], how='left')
stacked_results_appr["Approximate"]= True

#Compute the amount of token connections for approximate marginalization
stacked_results_appr["Token_Connections"]= (n_train ** 2) + (n_train * n_inf *  stacked_results_appr["M"] * stacked_results_appr["L"])


#Aggregate results of exact marginalization
stacked_results_exact_mean= exp_results["loss_exact_marginalization_mean"].stack().reset_index()
stacked_results_exact_mean.columns = ['M', 'L', 'Loss_Mean']

stacked_results_exact_std= exp_results["loss_exact_marginalization_std"].stack().reset_index()
stacked_results_exact_std.columns = ['M', 'L', 'Loss_Std']

stacked_results_exact= pd.merge(stacked_results_exact_mean, stacked_results_exact_std, on=['M', 'L'], how='left')
stacked_results_exact["Approximate"]= False

#Compute the amount of token connections for exact marginalization
stacked_results_exact["Token_Connections"]= ((n_train ** 2) * stacked_results_appr["M"]) + (n_train * n_inf * stacked_results_appr["M"])

#Combine results
stacked_results= pd.concat([stacked_results_appr, stacked_results_exact], axis= 0, ignore_index= True)

# plot_results_Conn_Perf()

def plot_results_TokenCons_Loss(exact_results,
                                 appr_results):
    plt.figure(figsize=(10, 6))

    # cmap_original = plt.get_cmap('magma')
    # cmap_inverted = ListedColormap(cmap_original.colors[::-1])

    plt.errorbar(x=exact_results['Token_Connections'], 
                y=exact_results['Loss_Mean'], 
                yerr=exact_results['Loss_Std'], 
                fmt='o', 
                linestyle='None', 
                capsize=0.1, 
                label= "Exact")
    
    plt.errorbar(x=appr_results['Token_Connections'], 
                y=appr_results['Loss_Mean'], 
                yerr=appr_results['Loss_Std'], 
                fmt='o', 
                linestyle='None', 
                capsize=0.01, 
                label= "Approximate")
    
    plt.title('Loss per Token Connections')
    plt.xlabel('Token Connections')
    plt.ylabel('Loss')
    plt.ylim(0, stacked_results['Loss_Mean'].max())
    plt.show()

plot_results_TokenCons_Loss(stacked_results_exact, stacked_results_appr)