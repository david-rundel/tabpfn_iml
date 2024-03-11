import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pickle
import openml

plt.rcParams.update({'font.size': 14})

"""
This file is used to generate the plots in section '4.3 Data valuation for Context Optimization' of the paper.
It processes the result files (in experiments/context_optimization/results) generated by running experiments/context_optimization/optimize_data_shapley.py.
"""

experiment_770_results_path = 'experiments/kernel_shap/results/kernel_shap_770_20240229_023201.pkl'
experiment_819_results_path = 'experiments/kernel_shap/results/kernel_shap_819_20240229_115943.pkl'
experiment_900_results_path = 'experiments/kernel_shap/results/kernel_shap_900_20240229_145322.pkl'

experiment_results_paths = [experiment_770_results_path,
                            experiment_819_results_path,
                            experiment_900_results_path]

exp_results = {}
for experiment_results_path in experiment_results_paths:
    with open(experiment_results_path, 'rb') as file:
        openml_id = int(experiment_results_path.split("_")[-3])
        exp_results[openml_id] = pickle.load(file)

cmap_original = plt.get_cmap('magma')
cmap_inverted = ListedColormap(cmap_original.colors[::-1])


def plot_error_per_ML(exp_results, save):
    plot_dfs = {}
    for openml_id, temp_exp_results in exp_results.items():
        plot_dfs[openml_id] = temp_exp_results["error_appr_marginalization_mean"].copy()
        plot_dfs[openml_id][-1] = temp_exp_results["error_exact_marginalization_mean"].copy()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    i = 0
    for openml_id, plot_df in plot_dfs.items():
        x, y = np.meshgrid(plot_df.columns, plot_df.index)
        x_flat = x.flatten()
        y_flat = y.flatten()
        values = plot_df.values.flatten()

        scatter = axs[i].scatter(
            x_flat, y_flat, c=values, cmap=cmap_inverted, s=15)
        axs[i].set_title(
            f'{str(openml.datasets.get_dataset(openml_id).name)} dataset')
        axs[i].set_xlabel('L')
        axs[i].set_ylabel('M')
        temp_xticks = [-1, 1]
        temp_xticks.extend(list(np.arange(5, max(plot_df.columns)+1, 5)))
        axs[i].set_xticks(temp_xticks)
        axs[i].axvline(x=0, color='gray', linestyle='-', linewidth=0.7)
        i += 1
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax, label='Error')

    if save:
        plt.savefig('experiments/kernel_shap/results/Error_M_L.pdf',
                    bbox_inches='tight')
    else:
        plt.show()


plot_error_per_ML(exp_results, save=False)


stacked_results = {}
for openml_id, temp_exp_results in exp_results.items():
    n_train = temp_exp_results["experiment_hyperparameters"]["n_train"]
    n_inf = temp_exp_results["experiment_hyperparameters"]["n_test"]

    # Aggregate results of approximate marginalization
    temp_stacked_results_appr_mean = temp_exp_results["error_appr_marginalization_mean"].stack(
    ).reset_index()
    temp_stacked_results_appr_mean.columns = ['M', 'L', 'Error_Mean']

    temp_stacked_results_appr_std = temp_exp_results["error_appr_marginalization_std"].stack(
    ).reset_index()
    temp_stacked_results_appr_std.columns = ['M', 'L', 'Error_Std']

    temp_stacked_results_appr = pd.merge(
        temp_stacked_results_appr_mean, temp_stacked_results_appr_std, on=['M', 'L'], how='left')
    temp_stacked_results_appr["Approximate"] = True

    # Compute the amount of token connections for approximate marginalization
    temp_stacked_results_appr["Token_Connections"] = (n_train ** 2) + (
        n_train * n_inf * temp_stacked_results_appr["M"] * temp_stacked_results_appr["L"])

    # Aggregate results of exact marginalization
    temp_stacked_results_exact_mean = temp_exp_results["error_exact_marginalization_mean"].stack(
    ).reset_index()
    temp_stacked_results_exact_mean.columns = ['M', 'L', 'Error_Mean']

    temp_stacked_results_exact_std = temp_exp_results["error_exact_marginalization_std"].stack(
    ).reset_index()
    temp_stacked_results_exact_std.columns = ['M', 'L', 'Error_Std']

    temp_stacked_results_exact = pd.merge(
        temp_stacked_results_exact_mean, temp_stacked_results_exact_std, on=['M', 'L'], how='left')
    temp_stacked_results_exact["Approximate"] = False

    # Compute the amount of token connections for exact marginalization
    temp_stacked_results_exact["Token_Connections"] = (
        (n_train ** 2) * temp_stacked_results_exact["M"]) + (n_train * n_inf * temp_stacked_results_exact["M"])

    # Combine results
    temp_stacked_results = pd.concat(
        [temp_stacked_results_appr, temp_stacked_results_exact], axis=0, ignore_index=True)

    stacked_results[openml_id] = [temp_stacked_results,
                                  temp_stacked_results_exact, temp_stacked_results_appr]


def plot_error_per_tc(stacked_results, save):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    quantiles = [0.97, 0.95, 0.999]
    i = 0

    for openml_id, temp_results in stacked_results.items():
        temp_stacked_results, temp_stacked_results_exact, temp_stacked_results_appr = temp_results

        axs[i].errorbar(x=temp_stacked_results_exact['Token_Connections'],
                        y=temp_stacked_results_exact['Error_Mean'],
                        yerr=temp_stacked_results_exact['Error_Std'],
                        fmt='none',
                        capsize=0.5,
                        elinewidth=1,
                        marker='_',
                        color=cmap_inverted.colors[100],
                        alpha=0.4)

        axs[i].scatter(temp_stacked_results_exact['Token_Connections'],
                       temp_stacked_results_exact['Error_Mean'],
                       s=2,
                       color=cmap_inverted.colors[100],
                       label="Exact retraining")

        axs[i].errorbar(x=temp_stacked_results_appr['Token_Connections'],
                        y=temp_stacked_results_appr['Error_Mean'],
                        yerr=temp_stacked_results_appr['Error_Std'],
                        fmt='none',
                        capsize=0.5,
                        elinewidth=1,
                        marker='_',
                        color=cmap_inverted.colors[230],
                        alpha=0.1)  # 0.05

        axs[i].scatter(temp_stacked_results_appr['Token_Connections'],
                       temp_stacked_results_appr['Error_Mean'],
                       s=2,
                       color=cmap_inverted.colors[230],
                       label="Appr. retraining")

        axs[i].set_title(
            f'{str(openml.datasets.get_dataset(openml_id).name)} (ID: {str(openml_id)})', fontsize=14)
        axs[i].set_xlabel('Token Connections', fontsize=14)
        axs[i].set_ylabel('Error', fontsize=14)
        axs[i].set_ylim(0, (temp_stacked_results['Error_Mean'] +
                        temp_stacked_results['Error_Std']).quantile(quantiles[i]))

        axs[i].set_xscale('log')

        i += 1

    plt.tight_layout()
    plt.legend(markerscale=5)

    if save:
        plt.savefig('experiments/kernel_shap/results/Error_TC.pdf',
                    bbox_inches='tight')
    else:
        plt.show()


plot_error_per_tc(stacked_results, save=True)
