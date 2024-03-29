import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import openml

plt.rcParams.update({'font.size': 14})

"""
This file is used to generate the plots in section '4.2 Kernel SHAP' of the paper.
It processes the result files (in experiments/kernel_shap/results) generated by running experiments/kernel_shap/benchmark.py.
"""

save = True
M = 9216
i = 0

cmap_original = plt.get_cmap('magma')
cmap_inverted = ListedColormap(cmap_original.colors[::-1])

fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=False)

openml_ids = [1471, 23512, 41147]
result_files_detailed = {1471: ["experiments/context_optimization/results/data_shapley_1471_20240309_120008.csv"],
                         23512: ["experiments/context_optimization/results/data_shapley_23512_20240309_220238.csv"],
                         41147: ["experiments/context_optimization/results/data_shapley_41147_20240310_083047.csv"]}

result_files_mean = {1471: ["experiments/context_optimization/results/data_shapley_mean_1471_20240309_120008.csv"],
                     23512: ["experiments/context_optimization/results/data_shapley_mean_23512_20240309_220238.csv"],
                     41147: ["experiments/context_optimization/results/data_shapley_mean_41147_20240310_083047.csv"]}

detailed_results = {}
mean_results = {}

for openml_id in openml_ids:
    detailed_results[openml_id] = pd.DataFrame()

    for file_path in result_files_detailed[openml_id]:
        detailed_results[openml_id] = pd.concat(
            [detailed_results[openml_id], pd.read_csv(file_path)], axis=0)

    mean_results[openml_id] = pd.DataFrame()

    for file_path in result_files_mean[openml_id]:
        mean_results[openml_id] = pd.concat(
            [mean_results[openml_id], pd.read_csv(file_path)], axis=0)

for openml_id in openml_ids:
    temp_mean_results = mean_results[openml_id].groupby(
        "M").agg("mean").reset_index()
    temp_mean_results = temp_mean_results[temp_mean_results["M"] == M][[
        "RC Mean ROC AUC", "OC ROC AUC"]]
    temp_mean_results["seed"] = "Mean"

    temp_detailed_results = detailed_results[openml_id][detailed_results[openml_id]["M"] == M]
    temp_detailed_results["seed"] = temp_detailed_results["seed"].astype(str)
    temp_detailed_results = temp_detailed_results[[
        "seed", "RC Mean ROC AUC", "OC ROC AUC"]]

    temp_results = temp_detailed_results

    temp_results["seed"] = list(range(1, 6))  # + ["Mean"]
    temp_results = temp_results.set_index("seed")

    temp_min = temp_detailed_results[[
        "RC Mean ROC AUC", "OC ROC AUC"]].min().min()
    temp_max = temp_detailed_results[[
        "RC Mean ROC AUC", "OC ROC AUC"]].max().max()
    temp_mid = (temp_min+temp_max)/2
    temp_plot_max = temp_mid + 3*(temp_max - temp_mid)
    temp_plot_min = temp_mid - 3*(temp_max - temp_mid)

    temp_results = temp_results.rename(columns={"RC Mean ROC AUC": "Random Context",
                                                "OC ROC AUC": "Optimized Context"})

    # axs[i].axhline(y=temp_mean_results["RC Mean ROC AUC"].item(), color=cmap_inverted.colors[230], linestyle='-', label='Horizontal Line')
    # axs[i].axhline(y=temp_mean_results["OC ROC AUC"].item(), color=cmap_inverted.colors[100], linestyle='-', label='Horizontal Line')

    temp_results.plot(kind="bar",
                      ax=axs[i],
                      ylim=(temp_plot_min, temp_plot_max),
                      legend=(True if i == 2 else False),
                      color=[cmap_inverted.colors[230],
                             cmap_inverted.colors[100]],
                      rot=0,
                      alpha=0.9,
                      fontsize=14)

    axs[i].set_title(
        f'{str(openml.datasets.get_dataset(openml_id).name)} (ID: {str(openml_id)})', fontsize=14)
    axs[i].set_xlabel("Run", fontsize=14)
    axs[i].set_ylabel("ROC AUC", fontsize=14)

    print("Difference ID: " + str(openml_id) + ":")
    print((temp_results["Optimized Context"] -
          temp_results["Random Context"]).mean())
    print("---")

    i += 1

plt.tight_layout()

if not save:
    plt.show()
else:
    plt.savefig(
        'experiments/context_optimization/results/ROCAUC_Run.pdf', bbox_inches='tight')
