import os
import sys

current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_dir)
module_dir = os.path.dirname(parent_dir)
sys.path.append(module_dir)

import numpy as np
from itertools import permutations

from tabpfniml.datasets.datasets import dataset_iml
from tabpfniml.methods.interpret import TabPFN_Interpret


class Shapley_Exact(TabPFN_Interpret):
    """
        Exact implementation of Shapley values.
        It is used in experiments/kernel_shap/benchmark.py.
    """

    def __init__(self,
                 data: dataset_iml,
                 n_train: int = 1024,
                 n_test: int = 512,
                 N_ensemble_configurations: int = 16,
                 device: str = "cpu",
                 debug: bool = False):
        """
        Initialize the exact Shapley value computation.

        Args:
            data (dataset_iml, optional): The dataset that TabPFN's behavior shall be explained for. Defaults to dataset_iml.
            n_train (int, optional): The amount of train-samples to fit TabPFN on. Should not be larger than 1024. Defaults to 512.
            n_test (int, optional): The amount of test-samples to get predictions for. Defaults to 512.
            N_ensemble_configurations (int, optional): The amount of TabPFN forward passes with different augmentations ensembled. Defaults to 16.
            device (str, optional): The device to store tensors and the TabPFN model on. Defaults to "cpu".
            debug (bool, optional): Whether debug mode is activated. This leads to e.g. less train and test samples and can hence tremendously reduce computational cost. Overwrites various other parameters. Defaults to False.
        """
        super().__init__(data=data,
                         n_train=n_train,
                         n_test=n_test,
                         N_ensemble_configurations=N_ensemble_configurations,
                         device=device,
                         debug=debug,
                         standardize_features=False,
                         to_torch_tensor=False,
                         store_gradients=False)

    def fit(self,
            class_to_be_explained: int = 1,
            debug=False
            ):
        """
        Computes exact Shapley values.
        For a dataset with p features, it requires p!*p forward passes.

        Args:
            class_to_be_explained (int, optional): The class that predictions are explained for. Defaults to 1.
            debug (bool, optional): If debug mode is activated, only 32 feature permutations are considered. Defaults to False.
        """
        self.class_to_be_explained = class_to_be_explained

        # Generate all p! feature permutations
        feature_indices_list = list(range(self.data.num_features))
        feature_permutations = list(permutations(feature_indices_list))

        perm_count = len(feature_permutations) if not debug else 32

        # Initialize dataframe to track marginal contributions of features
        marg_cont = np.zeros(
            (self.X_test.shape[0], self.data.num_features, perm_count))

        # Set relative frequency of class 1 as mean pred
        self.mean_pred = self.y_train.mean()

        if debug:
            feature_permutations = feature_permutations[:perm_count]

        for index, permutation in enumerate(feature_permutations):
            prev_pred_in_coalition = np.full(
                self.X_test.shape[0], self.mean_pred)

            # For each permutation and for all first k elements for k=1,...,p refit the model and compute
            # the difference in prediction to the subset of the first k-1 elements
            for permutation_index in range(len(permutation)):
                temp_features = permutation[:permutation_index+1]

                X_train_masked = self.X_train[:, temp_features].copy()
                X_test_masked = self.X_test[:, temp_features].copy()

                self.classifier.fit(X_train_masked, self.y_train.copy())
                temp_pred = self.classifier.predict_proba(
                    X_test_masked)[:, self.class_to_be_explained]

                marg_cont[:, temp_features[-1],
                          index] = np.array(temp_pred - prev_pred_in_coalition)
                prev_pred_in_coalition = temp_pred

        self.shapley_values = marg_cont.mean(axis=-1)
