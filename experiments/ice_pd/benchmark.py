import time
import numpy as np
import pandas as pd
import openml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from PyALE import ale
from tabpfn import TabPFNClassifier
import datetime
import torch

from tabpfniml.methods.ice_pd import ICE_PD
from tabpfniml.methods.ale import ALE
from tabpfniml.datasets.datasets import ArrayData, OpenMLData

if torch.backends.mps.is_available():
    DEVICE = 'mps'

elif torch.cuda.is_available():
    DEVICE = 'cuda'

else:
    DEVICE = 'cpu'

print(f"Using device: {DEVICE}")

N = 1000

N_TRAIN = round(0.8 * N)
N_TEST = round(0.2 * N)

FEATURE_ID_OF_INTEREST = 1

def generate_classification_dataset(seed: int, n:int = 1000, p:int = 10, levels:int = 10, feature:int = 0):
    # Step 1: Generate random features matrix X
    np.random.seed(seed)
    X = np.random.rand(n, p)
    
    # Step 2: Construct a nonlinear function of the features plus noise
    # Example nonlinear function: sum of sin of all features plus random noise
    noise = np.random.normal(0, 0.1, size=n)
    nonlinear_sum = np.sum(np.sin(X), axis=1) + np.sum(np.exp(X), axis=1) - np.sum(np.log(X + 1), axis=1)
    y_continuous = nonlinear_sum + noise

    # Disretize feature of interest, e.g., feature 0, into 'levels' classes:
    discretizer_x = KBinsDiscretizer(n_bins=levels, encode='ordinal', strategy='quantile')
    X[:, feature] = discretizer_x.fit_transform(X[:, feature].reshape(-1, 1)).flatten()
    
    # Step 3: Discretize y_continuous into 2 classes
    discretizer = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform')
    y = discretizer.fit_transform(y_continuous.reshape(-1, 1)).flatten().astype(int)
    
    return X, y


def benchmark_implementations(X_train, y_train, oml, feature, level, n_train, n_test):
    # Initialize classifier
    time_init_start = time.time()
    clf = TabPFNClassifier(seed=42)
    clf.fit(X_train, y_train)
    time_init = time.time() - time_init_start

    # scikit-learn PDP
    print("Scikit-learn PDP...")
    start_time_pdp = time.time()
    partial_dependence(clf, X_train[:n_train,:], features=[feature], grid_resolution=level, percentiles=(0, 1), method='brute')
    sklearn_runtime_pdp = time.time() - start_time_pdp
    print("Scikit-learn PDP done.")

    # PyALE ALE
    print("PyALE ALE...")
    start_time_ale = time.time()
    ale(pd.DataFrame(X_train[:n_train,:]), clf, [feature], grid_size=level, plot=False)
    pyale_runtime_ale = time.time() - start_time_ale
    print("PyALE ALE done.")

    # tabpfn_iml PDP Implementation
    print("tabpfn_iml PDP...")
    icepd = ICE_PD(oml, n_train=n_train, n_test=n_test)
    start_time_iml = time.time()
    icepd.fit(features=[feature], max_levels_per_feature=level)
    icepd.get_PD(feature=feature)
    tabpfniml_runtime_pdp = time.time() - start_time_iml
    print("tabpfn_iml PDP done.")

    # tabpfn_iml ALE Implementation
    print("tabpfn_iml ALE...")
    ale_obj = ALE(oml, n_train=n_train, n_test=n_test)
    start_time_iml_ale = time.time()
    ale_obj.fit(features=[feature], discretize_by_linear_spacing=True, center=False, max_intervals_per_feature=level)
    # ale_obj.get_ALE(feature=feature)
    tabpfniml_runtime_ale = time.time() - start_time_iml_ale
    print("tabpfn_iml ALE done.")

    return sklearn_runtime_pdp, tabpfniml_runtime_pdp, pyale_runtime_ale, tabpfniml_runtime_ale


# benchmark loop
def run_benchmark(feature_id_of_interest:int, openml: bool = False, open_cc_dids=None, level_list=None):
    if openml:
        results = []
        for did in open_cc_dids:
            for current_seed in [42, 43, 44, 45, 46]:
                print("=" * 50)
                print(f"Benchmarking dataset {did} and seed {current_seed}...")
                oml_temp = OpenMLData(did)
                print(f"Categorical feature indices: {oml_temp.categorical_features_idx}")
                n_train = round(0.8*oml_temp.X.shape[0])
                n_test = round(0.2*oml_temp.X.shape[0])
                # The following line does not make sense, for ALE, the comp. complexity depends on the number of feature intervals, not the number of unique feature values.
                sklearn_runtime_pdp, tabpfniml_runtime_pdp, pyale_runtime_ale, tabpfniml_runtime_ale = benchmark_implementations(X_train=oml_temp.X,
                                                                                                                                y_train=oml_temp.y,
                                                                                                                                oml=oml_temp,
                                                                                                                                feature=feature_id_of_interest,
                                                                                                                                level=100, 
                                                                                                                                n_train=n_train,
                                                                                                                                n_test=n_test)
                results.append({
                    "dataset": did,
                    "scikit-learn PDP": sklearn_runtime_pdp,
                    "tabpfn_iml PDP": tabpfniml_runtime_pdp,
                    "PyALE ALE": pyale_runtime_ale,
                    "tabpfn_iml ALE": tabpfniml_runtime_ale,
                    "Number of unique Featurevalues": np.unique(oml_temp.X[:n_train, feature_id_of_interest]).shape[0],
                    "seed": current_seed
                })

    else:
        results = []
        for level in level_list:
            for current_seed in [42, 43, 44, 45, 46]:
                print("=" * 50)
                print(f"Benchmarking dataset with levels {level} and seed {current_seed}...")
                X_train, y_train = generate_classification_dataset(levels=level, feature=feature_id_of_interest, seed=current_seed)
                oml_temp = ArrayData(str(level), X_train, y_train, categorical_features_idx=[], feature_names=[f'x{i}' for i in range(X_train.shape[1])])
                X_train = oml_temp.X
                print(X_train.shape)
                y_train = oml_temp.y
                print(y_train.shape)
                sklearn_runtime_pdp, tabpfniml_runtime_pdp, pyale_runtime_ale, tabpfniml_runtime_ale = benchmark_implementations(X_train=X_train,
                                                                                                                                y_train=y_train,
                                                                                                                                oml=oml_temp,
                                                                                                                                feature=feature_id_of_interest,
                                                                                                                                level=level,
                                                                                                                                n_train=N_TRAIN,
                                                                                                                                n_test=N_TEST)
                results.append({
                    "dataset": level,
                    "scikit-learn PDP": sklearn_runtime_pdp,
                    "tabpfn_iml PDP": tabpfniml_runtime_pdp,
                    "PyALE ALE": pyale_runtime_ale,
                    "tabpfn_iml ALE": tabpfniml_runtime_ale,
                    "Number of unique Featurevalues": np.unique(X_train[:, feature_id_of_interest]).shape[0],
                    "seed": current_seed
                })

    results_df = pd.DataFrame(results)

    results_df_pdp = results_df[["dataset", "Number of unique Featurevalues", "scikit-learn PDP", "tabpfn_iml PDP", "seed"]]
    results_df_ale = results_df[["dataset", "Number of unique Featurevalues", "PyALE ALE", "tabpfn_iml ALE", "seed"]]

    # Melt the sorted DataFrame for plotting with seaborn
    melted_df_pdp = results_df_pdp.melt(id_vars=["dataset", "Number of unique Featurevalues", "seed"],
                            var_name="Implementation", value_name="Runtime in Seconds")

    melted_df_ale = results_df_ale.melt(id_vars=["dataset", "Number of unique Featurevalues", "seed"],
                        var_name="Implementation", value_name="Runtime in Seconds")
    
    return melted_df_pdp, melted_df_ale


# melted_df_pdp, melted_df_ale = run_benchmark(openml=False, level_list=[5, 10, 25, 50, 100, 200, 300], feature_id_of_interest=1)
# current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
# melted_df_pdp.to_csv(f"./results/benchmark_results_pdp_{current_time}.csv")
# melted_df_ale.to_csv(f"./results/benchmark_results_ale_{current_time}.csv")
# print("Benchmarking synthetic datasets done.")

melted_df_pdp, melted_df_ale = run_benchmark(openml=True, open_cc_dids=[11, 15, 31, 1049, 37, 40982, 1494], feature_id_of_interest=1)
current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
melted_df_pdp.to_csv(f"./results/benchmark_results_pdp_{current_time}.csv")
melted_df_ale.to_csv(f"./results/benchmark_results_ale_{current_time}.csv")
print("Benchmarking synthetic datasets done.")
