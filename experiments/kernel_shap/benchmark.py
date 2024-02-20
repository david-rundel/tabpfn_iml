import sys
sys.path.append('/Users/davidrundel/git/tabpfn_iml/')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

debug= True
if debug:
    n_train= 4 #At least > p
    n_test= 2 #At least 2
    max_s= 2 #At least 2
    runs= 2 #At least 2

else:
    n_train= 512
    n_test= 256
    max_s= 4
    runs= 20

data= OpenMLData(770) #841

#Track the results of exact and approximate marginalization across multiple runs
loss_exact_marg_runs= {}
loss_appr_marg_runs= {}

#Track the feature subsets and imputation sample indices to ensure that various runs are different
design_matrix_runs= {}
weights_runs= {}
random_train_indices= {}

plot_each_run= True

#Compute exact Shapley values (ground truth)
#Since there is no stochasticity in Shapley value computation, we do not have to repeat it multiple times.
shapley_exact = Shapley_Exact(data= data, 
                                n_train= n_train,
                                n_test= n_test)
shapley_exact.fit(debug= debug)


for run in range(runs):
    #Compute KernelSHAP with approximate and exact marginalization
    #Stochasticity in results due to coalitions and approximate marginalization [LM fitting procedure does not introduce stochasticity due to seed]
    kernel_shap = Kernel_SHAP(data=data,
                                n_train= n_train,
                                n_test= n_test)
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

    loss_appr_marg_df= pd.DataFrame(columns= list(range(max_s)))
    for key, value in loss_appr_marg.items():
        loss_appr_marg_df.loc[key]= value

    loss_exact_marg_runs[run]= loss_exact_marg_df
    loss_appr_marg_runs[run]= loss_appr_marg_df
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
    scatter = plt.scatter(x_flat, y_flat, c=values, cmap='viridis', s=30)
    plt.colorbar(scatter, label='Values')
    plt.title('Loss per M and L')
    plt.xlabel('L')
    plt.ylabel('M')
    plt.show()

    del plot_df

if plot_each_run:
    for run in range(runs):
        plot_results(loss_appr_marg_runs[run],
                     loss_exact_marg_runs[run],
                     weights_runs[run])
        
#Average results across runs
loss_exact_marg_mean= pd.DataFrame(np.concatenate([loss_exact_marg_runs[i] for i in range(5)], axis=1).mean(axis=-1))
loss_appr_marg_mean= pd.DataFrame(np.stack([loss_appr_marg_runs[i] for i in range(5)]).mean(axis=0))
weights_mean= None #TODO

plot_results(loss_appr_marg_mean,
             loss_exact_marg_mean,
             weights_runs[run])








###############################################################################
# TODOs:
#     -Timen
#     -Andere Datensätze testen
#     -Disclaimer: Approximate Marg hier langsamer
#     -Dieses und andere Files aufräumen
#     -Variablennamen wie in paper

#bestimmte design matrix kann über alle plötzlich hohen effekt haben
#bestimmtes imputation auch (bisschen weniger)
#weight analysieren
#plotting hier einbauen

#solution: nicht nur mit einem training sample imputen sondern mehreren (so weniger sensitiv dagegen)
    #dann evtl in paper noch anpassen
#nur coalitions mit hohem weight? effektive amount of coals
###############################################################################

