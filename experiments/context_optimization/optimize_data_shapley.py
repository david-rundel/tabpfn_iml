import os
import sys

current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_dir)
module_dir = os.path.dirname(parent_dir)
sys.path.append(module_dir)

from tabpfniml.methods.data_shapley import Data_Shapley
import numpy as np
import pandas as pd
import datetime
import random
import torch
import pickle

if torch.backends.mps.is_available():
    DEVICE = 'mps'

elif torch.cuda.is_available():
    DEVICE = 'cuda'

else:
    DEVICE = 'cpu'

print(f"Using device: {DEVICE}")


debug = True
if debug:
    n_train = 32
    n_val = 8
    n_test = 8
    M_factor = 2
    tPFN_train_min = 8
    tPFN_train_max = 16
    runs = 2

else:
    n_train = 3072
    n_val = 512
    n_test = 1024
    M_factor = 3
    tPFN_train_min = 256
    tPFN_train_max = 512
    runs = 5

# Default 0. Can be used to proceed an experiment where there have already been seed_shift-many runs with different seeds.
seed_shift = 0


openml_ids = [1471, 23512, 41147]
for openml_id in openml_ids:
    # Ensure reproducibility of conducted experiments across several runs
    random.seed(42)
    seeds = [random.randint(1, 10000) for _ in range(
        seed_shift + runs)][seed_shift:seed_shift+runs]
    formatted_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment_results = pd.DataFrame()

    for seed in seeds:
        data_shapley = Data_Shapley(optimize_context=True,
                                    openml_id=openml_id,
                                    n_train=n_train,
                                    n_val=n_val,
                                    n_test=n_test,
                                    seed=seed
                                    )

        design_matrix, weights, loss_values = data_shapley.fit(M_factor=M_factor,
                                                               tPFN_train_min=tPFN_train_min,
                                                               tPFN_train_max=tPFN_train_max,
                                                               class_to_be_explained=1,
                                                               return_intermediate_results=True)

        design_matrix.to_csv('experiments/context_optimization/results/intermediate_results/design_matrix_' +
                             str(openml_id) + '_' + str(seed) + '_' + formatted_datetime + '.csv', index=True)
        weights.to_csv('experiments/context_optimization/results/intermediate_results/weights_' +
                       str(openml_id) + '_' + str(seed) + '_' + formatted_datetime + '.csv', index=True)
        loss_values.to_csv('experiments/context_optimization/results/intermediate_results/loss_values_' +
                           str(openml_id) + '_' + str(seed) + '_' + formatted_datetime + '.csv', index=True)

        data_values = data_shapley.get_data_values()
        opt_context = data_shapley.get_optimized_context()
        perf_diff_df = data_shapley.get_optimized_performance_diff()

        try:
            experiment_results = pd.concat(
                [experiment_results, perf_diff_df], axis=0)
        except:
            experiment_results = perf_diff_df

    experiment_results.to_csv('experiments/context_optimization/results/data_shapley_' +
                              str(openml_id) + '_' + formatted_datetime + '.csv', index=True)

    experiment_results_mean = experiment_results.drop(
        columns=["seed"]).groupby(["M"]).mean()
    experiment_results_mean.to_csv('experiments/context_optimization/results/data_shapley_mean_' +
                                   str(openml_id) + '_' + formatted_datetime + '.csv', index=True)

    hp_dict = {"n_train": n_train,
               "n_val": n_val,
               "n_test": n_test,
               "M_factor": M_factor,
               "tPFN_train_min": tPFN_train_min,
               "tPFN_train_max": tPFN_train_max,
               "runs": runs}

    with open('experiments/context_optimization/results/data_shapley_hps_' + str(openml_id) + '_' + formatted_datetime + '.pkl', 'wb') as file:
        pickle.dump(hp_dict, file)

    np.save('experiments/context_optimization/results/intermediate_results/X_test' +
            str(openml_id) + '_' + formatted_datetime + '.csv',
            data_shapley.X_test)

    np.save('experiments/context_optimization/results/intermediate_results/y_test' +
            str(openml_id) + '_' + formatted_datetime + '.csv',
            data_shapley.y_test)

    print(experiment_results)

print("Done.")
