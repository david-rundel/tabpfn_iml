import sys
sys.path.append('/Users/davidrundel/git/tabpfn_iml/') #Modify

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import datetime
import pickle
import random

from tabpfniml.datasets.datasets import OpenMLData
from experiments.kernel_shap.exact_shapley import Shapley_Exact
from experiments.kernel_shap.kernel_shap import Kernel_SHAP

"""
Python Script for the experiments conducted for the paper. We compare our approach to 
estimate Shapley values via Kernel SHAP in conjunction with TabPFN against a naive 
implementation that approximates marginalization of features that are absent in feature subsets.
We also implement the exact computation of Shapley values, in order to evaluate the accuracy of 
the different approximations.
As a consequence, due to bad scaling of the computation of the exact SHapley values, our benchmark
is restricted to datasets with a limited amount of features.

However, since we want to track several metrics and intermediate results, we do not use the versions
in the tabpfniml-folder. Instead, we use experiments/kernel_shap/kernel_shap.py and 
experiments/kernel_shap/shapley.py to compute the exact and approximate versions.
"""

#Set experiment specific HPs
debug= False
if debug:
    n_train= 64 #At least > p
    n_test= 16 #At least 2
    max_s= 4 #At least 2
    runs= 5 #At least 2

else:
    n_train= 256
    n_test= 64
    max_s= 16
    runs= 10

openml_id= 700
plot_each_run= False

#Ensure reproducibility of conducted experiments across several runs
random.seed(42)
seeds= [random.randint(1, 10000) for _ in range(runs)]

#Fetch data from OpenML
data= OpenMLData(openml_id) 

#Track the results of exact and approximate marginalization across multiple runs
loss_exact_marg_runs= {}
loss_appr_marg_runs= {}

kernel_shap_exact_marg_values_runs= {}
kernel_shap_appr_marg_values_runs= {}

#Track the feature subsets and imputation sample indices to ensure that various runs are different
design_matrix_runs= {}
weights_runs= {}
random_train_indices= {}

#Compute exact Shapley values (ground truth)
#Since there is no stochasticity in Shapley value computation, we do not have to repeat it multiple times.
shapley_exact = Shapley_Exact(data= data, 
                                n_train= n_train,
                                n_test= n_test)
shapley_exact.fit(debug= debug)


for run in range(runs):
    print("Starting run " + str(run) + ".")
    #Compute KernelSHAP with approximate and exact marginalization
    #Stochasticity in results due to coalitions and approximate marginalization [LM fitting procedure does not introduce stochasticity due to seed]
    kernel_shap = Kernel_SHAP(data=data,
                                n_train= n_train,
                                n_test= n_test,
                                run_seed= seeds[run])
    kernel_shap.fit(max_s= max_s)

    #Check if train and test subsets align
    np.testing.assert_array_equal(kernel_shap.X_train, shapley_exact.X_train)
    np.testing.assert_array_equal(kernel_shap.X_test, shapley_exact.X_test)

    #Compute the loss for varyiing amount of K (feature subsets) and s (imputation samples)
    loss_exact_marg= {}
    loss_appr_marg= {}

    for k_temp in kernel_shap.SHAP_exact_marg.keys():
        loss_exact_marg[k_temp]= np.mean(np.sum(np.abs(shapley_exact.shapley_values- kernel_shap.SHAP_exact_marg[k_temp].to_numpy()[:,1:]), axis= 1)) #Do not consider intercept in loss

    for (k_temp,s_temp) in kernel_shap.SHAP_approximate_marg.keys():
        loss_appr_marg[k_temp]= {}
    for (k_temp,s_temp) in kernel_shap.SHAP_approximate_marg.keys():
        loss_appr_marg[k_temp][s_temp]= np.mean(np.sum(np.abs(shapley_exact.shapley_values- kernel_shap.SHAP_approximate_marg[(k_temp,s_temp)].to_numpy()[:,1:]), axis= 1)) #Do not consider intercept in loss

    loss_exact_marg_df= pd.DataFrame.from_dict(loss_exact_marg, orient='index')

    loss_appr_marg_df= pd.DataFrame(columns= list(range(1, max_s+1)))
    for key, value in loss_appr_marg.items():
        loss_appr_marg_df.loc[key]= value

    loss_exact_marg_runs[run]= loss_exact_marg_df
    loss_appr_marg_runs[run]= loss_appr_marg_df

    kernel_shap_exact_marg_values_runs[run]= kernel_shap.SHAP_exact_marg
    kernel_shap_appr_marg_values_runs[run]= kernel_shap.SHAP_approximate_marg

    design_matrix_runs[run]= kernel_shap.design_matrix
    weights_runs[run]= kernel_shap.weights
    random_train_indices[run]= kernel_shap.random_train_indices

    del kernel_shap
    del loss_exact_marg_df
    del loss_appr_marg_df


def plot_results(loss_appr_marg,
                 loss_exact_marg,
                 weights):
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

#Plot results per run
if plot_each_run:
    for run in range(runs):
        plot_results(loss_appr_marg_runs[run],
                     loss_exact_marg_runs[run],
                     weights_runs[run])
        
#Aggregate results across runs
appr_cols= list(range(1, max_s+1))
exact_col= [-1]
loss_exact_marg_mean= pd.DataFrame(np.concatenate([loss_exact_marg_runs[i] for i in range(runs)], axis=1).mean(axis=-1), columns= exact_col)
loss_appr_marg_mean= pd.DataFrame(np.stack([loss_appr_marg_runs[i] for i in range(runs)]).mean(axis=0), columns= appr_cols)
loss_exact_marg_std= pd.DataFrame(np.concatenate([loss_exact_marg_runs[i] for i in range(runs)], axis=1).std(axis=-1), columns= exact_col)
loss_appr_marg_std= pd.DataFrame(np.stack([loss_appr_marg_runs[i] for i in range(runs)]).std(axis=0), columns= appr_cols)
weights_mean= None #TODO

#Save results
experiment_results= {"loss_exact_marginalization_per_run": loss_exact_marg_runs,
                        "loss_appr_marginalization_per_run": loss_appr_marg_runs,

                        "loss_exact_marginalization_mean": loss_exact_marg_mean,
                        "loss_appr_marginalization_mean": loss_appr_marg_mean,
                        "loss_exact_marginalization_std": loss_exact_marg_std,
                        "loss_appr_marginalization_std": loss_appr_marg_std,

                        "exact_shapley_values": shapley_exact.shapley_values,
                        "kernel_shap_values_exact_marg_runs": kernel_shap_exact_marg_values_runs,
                        "kernel_shap_values_appr_marg_runs": kernel_shap_appr_marg_values_runs,

                        "design_matrix_per_run": design_matrix_runs, 
                        "weights_per_run": weights_runs, 
                        "train_indices_per_run": random_train_indices,
                        "experiment_hyperparameters": {"n_train": n_train, 
                                                       "n_test": n_test, 
                                                       "max_s": max_s, 
                                                       "debug": debug,
                                                       "runs": 10, 
                                                       "openml_id": openml_id,
                                                       "seeds": seeds}
                        }

formatted_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

with open('experiments/kernel_shap/results/kernel_shap_' + str(openml_id) + '_' + formatted_datetime + '.pkl', 'wb') as file:
    pickle.dump(experiment_results, file)

#Plot overall results
plot_results(loss_appr_marg_mean,
             loss_exact_marg_mean,
             weights_runs[run])

print("Experiment done.")

# with open('experiments/kernel_shap/results/....pkl', 'rb') as file:
#     exp_results = pickle.load(file)

