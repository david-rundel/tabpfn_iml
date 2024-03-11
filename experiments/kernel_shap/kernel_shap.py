import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.special import comb
import random

from tabpfniml.datasets.datasets import dataset_iml
from tabpfniml.methods.interpret import TabPFN_Interpret

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Kernel_SHAP(TabPFN_Interpret):
    """
    Implementation of approximate and exact retraining/ feature marginalization for Kernel SHAP used in experiments/kernel_shap/benchmark.py.
    It is a modified version of 'tabpfniml/methods/kernel_shap.py', but more efficient when computing 
    exact and approximate feature marginalization across multiple values of M and L simultaneously.
    Also, it is limited to feature effect measures.
    """

    def __init__(self,
                 data: dataset_iml,
                 n_train: int = 1024,
                 n_test: int = 512,
                 N_ensemble_configurations: int = 16,
                 device: str = "cpu",
                 debug: bool = False,
                 run_seed: int = 728):
        """
        Initialize the Kernel SHAP object for the experiments.

        Args:
            data (dataset_iml, optional): The dataset that TabPFN's behavior shall be explained for. Defaults to dataset_iml.
            n_train (int, optional): The amount of train-samples to fit TabPFN on. Should not be larger than 1024. Defaults to 512.
            n_test (int, optional): The amount of test-samples to get explanations for. Defaults to 512.
            N_ensemble_configurations (int, optional): The amount of TabPFN forward passes with different augmentations ensembled. Defaults to 16.
            device (str, optional): The device to store tensors and the TabPFN model on. Defaults to "cpu".
            debug (bool, optional): Whether debug mode is activated. This leads to e.g. less train and test samples and can hence tremendously reduce computational cost. Overwrites various other parameters. Defaults to False.
            run_seed (int, optional): A seed that is set for each run to enable reproducibility. Defaults to 728.
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

        # Ensure stochasticity across several averaged runs (different feature subsets and imputation samples)
        random.seed(run_seed)
        np.random.seed(run_seed)

    def fit(self,
            class_to_be_explained: int = 1,
            max_L: int = 16
            ):
        """
        Fits Kernel SHAP (as feature effect measure) with exact and approximate feature marginalizations for several values of M and L simultaneously.

        Args:
            class_to_be_explained (int, optional): The class that predictions are explained for. Defaults to 1.
            max_L (int, optional): The maximum amount of imputation samples to use. All values for L from 1 to max_L will be evaluated. Defaults to 16.
        """
        def get_kernel_weight(cs: int, p: int) -> np.float64:
            """
            Compute weights for coalitions through the kernel.
            Small and large coalitions get the highest weights.

            Args:
                cs (int): The amount of features in the coalition.
                p (int): The total amount of features.

            Returns:
                np.float64: The weight for a coalition of this size.
            """
            if cs == 0:
                return 0
                # Better for usage in WLS than inf
            if cs == p:
                return 0
                # Better for usage in WLS than inf
            return (p - 1) / (comb(p, cs) * cs * (p - cs))

        self.class_to_be_explained = class_to_be_explained

        # Specify the sizes of feature subsets that we test
        self.M_min = self.data.num_features  # To avoid n<p issues
        # Sample a greater number of coalitions than the total count of distinct ones, as not every unique coalition may be included in the sampling process otherwise.
        self.M_max_approximate = int(2 ** self.data.num_features * 2.5)
        self.M_max_exact = self.M_max_approximate * 10

        self.design_matrix = pd.DataFrame()
        self.weights = pd.Series()  # (K)

        pred_values_exact_marg = pd.DataFrame()
        pred_values_approximate_marg = {}
        for j in range(1, max_L+1):
            pred_values_approximate_marg[j] = pd.DataFrame()

        def get_coal():
            coalition_mask = np.random.binomial(
                n=1, p=0.5, size=self.data.num_features)
            coalition_mask_bool = np.array(coalition_mask, dtype=bool)
            return coalition_mask, coalition_mask_bool

        for m in range(self.M_max_exact):
            # Step 1: Sample K coalitions
            coalition_mask, coalition_mask_bool = get_coal()

            # Case that only falses are sampled
            while sum(coalition_mask) == 0:
                coalition_mask, coalition_mask_bool = get_coal()

            self.design_matrix = pd.concat([self.design_matrix, pd.DataFrame([coalition_mask])],
                                           ignore_index=True,
                                           axis=0)

            # Step 3: Compute weights through Kernel
            weight = get_kernel_weight(
                coalition_mask.sum(), self.data.num_features)
            self.weights = pd.concat([self.weights, pd.Series(weight)],
                                     ignore_index=True,
                                     axis=0)

            # Step 2: Transfer Coalitions into feature space and get predictions by applying ML model.
            # For TabPFN there is no need to impute the features that do not appear in the coalition by sampling from the marginal distribution.
            # Instead they can simply be left out and one can refit TabPFN yielding an exact solution.

            # Obtain data for exact marginalization
            X_train_masked = self.X_train[:, coalition_mask_bool].copy()
            X_test_masked = self.X_test[:, coalition_mask_bool].copy()

            self.classifier.fit(X_train_masked, self.y_train.copy())
            preds_exact = self.classifier.predict_proba(
                X_test_masked)[:, self.class_to_be_explained]

            pred_values_exact_marg = pd.concat([pred_values_exact_marg,
                                                pd.DataFrame(preds_exact).transpose()],
                                               ignore_index=True,
                                               axis=0)

            # Obtain data for approximate marginalization
            if m < self.M_max_approximate:
                # In approximate case TabPFN only needs to be fit once, since training data and features remain constant.
                # (Although the self-attention computation seems to be repeated every time.)
                self.classifier.fit(self.X_train, self.y_train)

                # Marginalize out non-coalition features by taking L samples from dataset to impute values
                preds_approximate = None

                # Iterate over MC-imputation-samples and average predicitons
                self.random_train_indices = []
                for j in range(max_L):
                    X_test_imputed = self.X_test.copy()
                    random_train_index = random.randint(
                        0, self.X_train.shape[0]-1)
                    X_test_imputed[:, ~coalition_mask_bool] = self.X_train[random_train_index,
                                                                           ~coalition_mask_bool].copy()  # Broadcast
                    temp_preds = self.classifier.predict_proba(
                        X_test_imputed)[:, self.class_to_be_explained]  # , return_logits= True

                    self.random_train_indices.append(random_train_index)

                    if preds_approximate is not None:
                        preds_approximate = np.column_stack(
                            (preds_approximate, temp_preds))

                    else:
                        preds_approximate = temp_preds

                for j in range(max_L):
                    # Compute the average prediction for the first j imputing samples
                    temp_mean = preds_approximate[:, :j+1].copy().mean(axis=1)

                    # pred_values_approximate_marg[j+1]= pred_values_approximate_marg[j+1].append(
                    #     pd.Series(temp_mean), ignore_index=True)

                    pred_values_approximate_marg[j+1] = pd.concat([pred_values_approximate_marg[j+1],
                                                                   pd.Series(temp_mean).to_frame().T],
                                                                  ignore_index=True,
                                                                  axis=0)

        def get_SHAP_values(temp_preds, K):
            # Step 4: Fit a weighted linear model
            column_names = ["intercept"] + list(self.data.feature_names)

            # Sklearn implementation well suited for multi-label-regression for SHAP values
            lm = LinearRegression()
            # Only consider first k coalitions
            lm.fit(X=self.design_matrix.iloc[:K, :],
                   y=temp_preds.iloc[:K, :],
                   sample_weight=self.weights.iloc[:K])

            self.SHAP_local_values = lm.coef_
            self.SHAP_local_intercept = lm.intercept_

            return pd.DataFrame(np.concatenate([np.expand_dims(self.SHAP_local_intercept, axis=1), self.SHAP_local_values], axis=1),
                                index=[
                                    "SHAP_local_obs_" + str(i+1) for i in range(self.SHAP_local_intercept.shape[0])],
                                columns=column_names)

        # Compute Kernel SHAP for exact marginalization
        self.SHAP_exact_marg = {}
        for m in range(self.M_min, self.M_max_exact+1):
            self.SHAP_exact_marg[m] = get_SHAP_values(
                pred_values_exact_marg, K=m)

        # Compute Kernel SHAP for approximate marginalization
        self.SHAP_approximate_marg = {}
        for m in range(self.M_min, self.M_max_approximate+1):
            for j in range(1, max_L+1):
                self.SHAP_approximate_marg[(m, j)] = get_SHAP_values(
                    pred_values_approximate_marg[j], K=m)
