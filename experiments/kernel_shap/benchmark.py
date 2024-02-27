# TO DELETE
import sys
# sys.path.append(
#     '/Users/juli/Documents/SoSe_23/Consulting/publication/tabpfn_iml')
sys.path.append('/Users/davidrundel/git/tabpfn_iml/')

from experiments.kernel_shap.kernel_shap import Kernel_SHAP
from experiments.kernel_shap.exact_shapley import Shapley_Exact
from tabpfniml.datasets.datasets import OpenMLData
import torch
import random
import pickle
import datetime
import pandas as pd
import numpy as np

"""
Python script to conduct the numerical experiments outlined in section '4.2 Kernel SHAP' of the paper.
The focus is on comparing our proposed method, Exact Feature Marginalization for TabPFN, with the classical 
approach introduced in the literature, namely, Approximate Feature Marginalization.

For the comparison, we design an error metric that measures the deviation from the exact Shapley values, 
as implemented in experiments/kernel_shap/exact_shapley.py. We do not resort to 'tabpfniml/methods/kernel_shap.py', 
but utilize 'experiments/kernel_shap/kernel_shap.py'. Thereby we modify it to compute exact and approximate feature 
marginalization across multiple values of M and L, W simultaneously.
"""

if torch.backends.mps.is_available():
    DEVICE = 'mps'

elif torch.cuda.is_available():
    DEVICE = 'cuda'

else:
    DEVICE = 'cpu'

print(f"Using device: {DEVICE}")

# Set experiment specific HPs
debug = False
if debug:
    n_train = 8  # At least > p
    n_test = 8  # At least 2
    max_L = 2  # At least 2
    runs = 2  # At least 2

else:
    n_train = 256
    n_test = 128
    max_L = 25
    runs = 25

openml_id = 819  # Considered datasets in paper: 770, 819, 900

# Ensure reproducibility of conducted experiments across several runs
random.seed(42)
seeds = [random.randint(1, 10000) for _ in range(runs)]

# Fetch data from OpenML
data = OpenMLData(openml_id)

# Track the results of exact and approximate marginalization across multiple runs
error_exact_marg_runs = {}
error_appr_marg_runs = {}
kernel_shap_exact_marg_values_runs = {}
kernel_shap_appr_marg_values_runs = {}

# Track the feature subsets and imputation sample indices to ensure that various runs are different
design_matrix_runs = {}
weights_runs = {}
random_train_indices = {}

# Compute exact Shapley values
# Since there is no stochasticity in Shapley value computation, we do not have to repeat it multiple times.
shapley_exact = Shapley_Exact(data=data,
                              n_train=n_train,
                              n_test=n_test,
                              device=DEVICE)
shapley_exact.fit(debug=debug)

for run in range(runs):
    print("Starting run " + str(run) + ".")
    # Compute KernelSHAP with approximate and exact feature marginalization.
    # Stochasticity in results due to sampling coalitions and approximate feature marginalization [LM fitting procedure does not introduce stochasticity due to seed].
    kernel_shap = Kernel_SHAP(data=data,
                              n_train=n_train,
                              n_test=n_test,
                              run_seed=seeds[run],
                              device=DEVICE)
    kernel_shap.fit(max_L=max_L)

    # Check if train and test subsets align
    np.testing.assert_array_equal(kernel_shap.X_train, shapley_exact.X_train)
    np.testing.assert_array_equal(kernel_shap.X_test, shapley_exact.X_test)

    # Compute the error for varying amount of M (feature subsets) and L (imputation samples)
    error_exact_marg = {}
    error_appr_marg = {}

    for M_temp in kernel_shap.SHAP_exact_marg.keys():
        error_exact_marg[M_temp] = np.mean(np.sum(np.abs(
            shapley_exact.shapley_values - kernel_shap.SHAP_exact_marg[M_temp].to_numpy()[:, 1:]), axis=1))  # Do not consider intercept in error

    for (M_temp, L_temp) in kernel_shap.SHAP_approximate_marg.keys():
        error_appr_marg[M_temp] = {}
    for (M_temp, L_temp) in kernel_shap.SHAP_approximate_marg.keys():
        error_appr_marg[M_temp][L_temp] = np.mean(np.sum(np.abs(shapley_exact.shapley_values - kernel_shap.SHAP_approximate_marg[(
            M_temp, L_temp)].to_numpy()[:, 1:]), axis=1))  # Do not consider intercept in error

    error_exact_marg_df = pd.DataFrame.from_dict(
        error_exact_marg, orient='index').round(3)

    error_appr_marg_df = pd.DataFrame(columns=list(range(1, max_L+1))).round(3)
    for key, value in error_appr_marg.items():
        error_appr_marg_df.loc[key] = value

    error_exact_marg_runs[run] = error_exact_marg_df
    error_appr_marg_runs[run] = error_appr_marg_df

    kernel_shap_exact_marg_values_runs[run] = kernel_shap.SHAP_exact_marg
    kernel_shap_appr_marg_values_runs[run] = kernel_shap.SHAP_approximate_marg

    design_matrix_runs[run] = kernel_shap.design_matrix
    weights_runs[run] = kernel_shap.weights
    random_train_indices[run] = kernel_shap.random_train_indices

    del kernel_shap
    del error_exact_marg_df
    del error_appr_marg_df


# Aggregate results across runs
appr_cols = list(range(1, max_L+1))
exact_col = [-1]
error_exact_marg_mean = pd.DataFrame(np.concatenate([error_exact_marg_runs[i] for i in range(
    runs)], axis=1).mean(axis=-1), columns=exact_col, index=error_exact_marg_runs[0].index).round(3)
error_appr_marg_mean = pd.DataFrame(np.stack([error_appr_marg_runs[i] for i in range(
    runs)]).mean(axis=0), columns=appr_cols, index=error_appr_marg_runs[0].index).round(3)
error_exact_marg_std = pd.DataFrame(np.concatenate([error_exact_marg_runs[i] for i in range(
    runs)], axis=1).std(axis=-1), columns=exact_col, index=error_exact_marg_runs[0].index).round(3)
error_appr_marg_std = pd.DataFrame(np.stack([error_appr_marg_runs[i] for i in range(
    runs)]).std(axis=0), columns=appr_cols, index=error_appr_marg_runs[0].index).round(3)

# Save results
experiment_results = {"error_exact_marginalization_per_run": error_exact_marg_runs,
                      "error_appr_marginalization_per_run": error_appr_marg_runs,

                      "error_exact_marginalization_mean": error_exact_marg_mean,
                      "error_appr_marginalization_mean": error_appr_marg_mean,
                      "error_exact_marginalization_std": error_exact_marg_std,
                      "error_appr_marginalization_std": error_appr_marg_std,

                      "exact_shapley_values": shapley_exact.shapley_values,
                      "kernel_shap_values_exact_marg_runs": kernel_shap_exact_marg_values_runs,
                      "kernel_shap_values_appr_marg_runs": kernel_shap_appr_marg_values_runs,

                      "design_matrix_per_run": design_matrix_runs,
                      "weights_per_run": weights_runs,
                      "train_indices_per_run": random_train_indices,
                      "experiment_hyperparameters": {"n_train": n_train,
                                                     "n_test": n_test,
                                                     "max_s": max_L,
                                                     "debug": debug,
                                                     "runs": runs,
                                                     "openml_id": openml_id,
                                                     "seeds": seeds}
                      }

formatted_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

with open('experiments/kernel_shap/results/kernel_shap_' + str(openml_id) + '_' + formatted_datetime + '.pkl', 'wb') as file:
    pickle.dump(experiment_results, file)

print("Experiment done.")
