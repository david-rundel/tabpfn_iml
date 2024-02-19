import pandas as pd
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from scipy.special import comb
import statsmodels.api as sm
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from tabpfniml.methods.interpret import TabPFN_Interpret
from tabpfniml.datasets.datasets import dataset_iml
from typing import Tuple, Dict, Optional, Union
import pickle
import os


class SHAP(TabPFN_Interpret):
    """
    Implementation of the Kernel-SHAP-method based on https://slds-lmu.github.io/iml/chapters/04_shapley/04_03_shap/.
    Kernel SHAP is a model agnostic method used for explaining model predictions. 
    It estimates the Shapley values, which assign each feature's contribution to the prediction based on its interaction with other features.
    In this implementation, SHAP values explain the difference from the model prediction when using no features at all (this corresponds to the 
    OCM with the respective loss - e.g. the fraction of observations with the class to be explained as label in the training data for the CE-Loss).

    Kernel SHAP is way more efficient than regular Shapley, since for every coalition the effect of all features in the coalition on the model
    prediction is estimated and not just the surplus contribution of one feature.

    Compared to the SHAP-package from Scott Lundberg (iml/methods/shap_package.py, https://shap.readthedocs.io/en/latest/index.html) it does not offer 
    a big variety of plots, however it is designed to be more exact and performant in combination with TabPFN.
    While the SHAP-package predicts for each test sample and feature coalition separately, this implementation iterates over feature coalitions 
    and predicts for all test samples at once.

    TODO: 
    -Explanations in log-odds space.
    """

    def __init__(self,
                 data: dataset_iml,
                 n_train: int = 1024,
                 n_test: int = 512,
                 N_ensemble_configurations: int = 16,
                 device: str = "cpu",
                 debug: bool = False):
        """
        Initialize a TabPFN-interpretability method by passing rudimentary objects and the general configuration that is constant across all flavors of the interpretability-method.
        For some variables it may be permitted to overwrite them in fit()-methods, although this is not intended in general.

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
            pred_based: bool = True,
            loss_based: bool = False,
            class_to_be_explained: int = 1,
            K: int = 512,
            apply_WLS: bool = True,
            n_test: Optional[int] = None
            ):
        """
        Fits a local surrogate model (weighted linear model) to predict model outputs for test samples given feature coalitions.
        The resulting surrogate model coefficients are interpreted as local Shapley values.
        A multi-label-regression is applied, where each target corresponds to one test sample (self.n_test many), K feature coalitions are used 
        as observations to estimate the effect of features on the model predictions.

        Local, absolute Shapley values can be averaged over test samples to estimate global SHAP as feature importance measure. 

        The effect of features on the test set performance can be estimated to obtain SAGE values. 
        In this case, the feature importance with respect to the whole test set is estimated and only single-label-regression is applied.

        Args:
            pred_based (bool, optional): Whether SHAP values (explaining model predictions) should be estimated. Defaults to True.
            loss_based (bool, optional): Whether SAGE values (explaining model performance) should be estimated. Defaults to False.
            class_to_be_explained (int, optional): The class that predictions are explained for. Ignored it pred_based=False. Defaults to 1.
            K (int, optional): Amount of coalitions to be sampled. Takes the role of amount of observations (n) in regression. To avoid n<p-problem ensure that it is greater than or equal to self.data.num_features. Enhance K to get more accurate estimates of local Shapley values. Defaults to 512.
            apply_WLS (bool, optional): Whether to apply WLS with the kernel-function or whether to sample coalitions based on the kernel-function and use OLS. Defaults to True.
            n_test (Optional[int], optional): Adapt the amount of test samples previously specified in __init__. This can be done to obtain more accurate global SHAP values (by measuring the effect of features w.r.t more test samples). Defaults to None.
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

        self.pred_based = pred_based
        self.loss_based = loss_based
        self.class_to_be_explained = class_to_be_explained
        self.K = K
        self.apply_WLS = apply_WLS

        if self.debug:
            # Must still be greater than or equal to self.data.num_features in order to avoid n<p-problem in OLS.
            self.K = 32

        if self.loss_based:
            self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        if n_test is not None:
            # Reload the data with adapted amount of test samples.
            # This can be done to obtain more accurate global SHAP values
            # (by measuring the effect of features w.r.t more test samples).
            self.X_train, self.X_test, self.y_train, self.y_test = self.data.load(n_train=self.n_train,
                                                                                  n_test=n_test)
            self.n_test = n_test

        design_matrix = pd.DataFrame()
        weights = pd.Series()  # (K)

        # Values to be explained (Test-set loss (SAGE) or per test-observation prediction (SHAP))
        if self.pred_based:
            pred_values = pd.DataFrame()
        if self.loss_based:
            loss_values = pd.DataFrame()

        if not self.apply_WLS:
            sum_kernel_shap_weights = np.sum([get_kernel_weight(
                i, self.data.num_features) for i in range(1, self.data.num_features)])
            weight_prob_dist = [(i, get_kernel_weight(i, self.data.num_features) /
                                 sum_kernel_shap_weights) for i in range(1, self.data.num_features)]

        for i in range(self.K):
            # Step 1: Sample K coalitions
            if self.apply_WLS:
                coalition_mask = np.random.binomial(
                    n=1, p=0.5, size=self.data.num_features)
                coalition_mask_bool = np.array(coalition_mask, dtype=bool)
            else:
                # Sample coalitions w.r.t probability distribution that has probabilities proportional to the kernel weights
                # and use OLS instead of WLS.
                coalition_features = np.random.choice([i[0] for i in weight_prob_dist], p=[
                                                      i[1] for i in weight_prob_dist])
                coalition_mask = np.array([1 for i in range(
                    coalition_features)] + [0 for i in range(self.data.num_features-coalition_features)])
                np.random.shuffle(coalition_mask)
                coalition_mask_bool = np.array(coalition_mask, dtype=bool)

            # Step 2: Transfer Coalitions into feature space and get predictions by applying ML model.
            # For TabPFN there is no need to impute the features that do not appear in the coalition by sampling from the marginal distribution.
            # Instead they can simply be left out and one can refit TabPFN yielding an exact solution.
            X_train_masked = self.X_train[:, coalition_mask_bool]
            X_test_masked = self.X_test[:, coalition_mask_bool]

            self.classifier.fit(X_train_masked, self.y_train)
            preds = self.classifier.predict_proba(X_test_masked)

            design_matrix = design_matrix.append(
                pd.Series(coalition_mask), ignore_index=True)

            if self.pred_based:
                pred_values = pred_values.append(
                    pd.Series(preds[:, self.class_to_be_explained]), ignore_index=True)
            if self.loss_based:
                loss = self.criterion(torch.tensor(preds), torch.tensor(
                    self.y_test, dtype=torch.long))  # .detach().numpy()
                loss_values = loss_values.append(
                    pd.Series(loss.item()), ignore_index=True)

            # Step 3: Compute weights through Kernel
            if self.apply_WLS:
                weight = get_kernel_weight(
                    coalition_mask.sum(), self.data.num_features)
                weights = weights.append(pd.Series(weight), ignore_index=True)

        # Step 4: Fit a weighted linear model (or linear model)
        column_names = ["intercept"] + list(self.data.feature_names)

        if self.pred_based:
            # Sklearn implementation well suited for multi-label-regression for SHAP values
            lm = LinearRegression()
            if self.apply_WLS:
                lm.fit(X=design_matrix, y=pred_values,
                       sample_weight=weights)
            else:
                lm.fit(X=design_matrix, y=pred_values)

            # Step 5: Extract SHAP values
            # Shapley values (self.n_test, self.data.num_features)
            self.SHAP_local_values = lm.coef_
            self.SHAP_local_intercept = lm.intercept_

            self.SHAP_global_values = np.mean(
                np.abs(lm.coef_), axis=0)  # (, num_features)
            self.SHAP_global_intercept = np.array(
                [np.mean(np.abs(lm.intercept_))])

            self.SHAP_local = pd.DataFrame(np.concatenate([np.expand_dims(self.SHAP_local_intercept, axis=1), self.SHAP_local_values], axis=1),
                                           index=[
                                               "SHAP_local_obs_" + str(i+1) for i in range(self.SHAP_local_intercept.shape[0])],
                                           columns=column_names)

            self.SHAP_global = pd.DataFrame(np.expand_dims(np.concatenate([self.SHAP_global_intercept, self.SHAP_global_values]), axis=0),
                                            index=["SHAP_global_values"],
                                            columns=column_names)

        if self.loss_based:
            # Statsmodels implementation better suited to extract test-statistics for SAGE values
            # SAGE values does not require multi-label-regression
            design_matrix = sm.add_constant(design_matrix)
            if self.apply_WLS:
                lm = sm.WLS(loss_values, design_matrix, weights=weights)
                self.lm_res = lm.fit()

            else:
                lm = sm.OLS(loss_values, design_matrix)
                self.lm_res = lm.fit()

            # Step 5: Extract SAGE values
            # Equivalent to SAGE values (, num_features)
            self.SAGE_global_values = np.array(self.lm_res.params[1:])
            self.SAGE_global_intercept = np.array([self.lm_res.params[0]])

            self.SAGE_global = pd.DataFrame(np.expand_dims(np.concatenate([self.SAGE_global_intercept, self.SAGE_global_values]), axis=0),
                                            index=["SAGE_values"],
                                            columns=column_names)

    def get_SHAP_values(self,
                        local: bool = False,
                        save_to_path: Optional[str] = None) -> Union[pd.Series, pd.DataFrame]:
        """
        Returns local or global SHAP values estimated in the fit()-function.
        SHAP values estimate the Shapley values, which assign each feature's contribution to the prediction based on its interaction with other features.
        Positive local SHAP values indicate how much higher the prediction is due too the feature while negative local SHAP values inicate how much smaller it is.
        Global SHAP values are always positive and indicate the influence of a feature on the prediction. 

        Args:
            local (bool, optional): Whether to return local SHAP values (per test sample) or the absolute values averaged over test samples (global SHAP as feature importance measure). Defaults to False.
            save_to_path (Optional[str], optional): If provided, save the dataframe to the specified path. Should end with '.csv'. Defaults to None.

        Raises:
            Exception: If the specified path to save the results does not work.
            Exception: If fit() has been applied with loss_based=True instead of loss_based=False or not at all.

        Returns:
            Union[pd.Series, pd.DataFrame]: Dataframe or Series of local or global SHAP values where columns correspond to features (plus an intercept that corresponds to the average prediction with empy feature coalition) 
            and rows to test observations that are being explained (only one for global SHAP if local=False).
        """
        try:
            if local:
                if save_to_path is not None:
                    try:
                        if not os.path.exists(os.path.dirname(save_to_path)):
                            os.makedirs(os.path.dirname(save_to_path))
                        self.SHAP_local.to_csv(save_to_path)
                    except:
                        raise ValueError(
                            "The specified path does not work. The path should end with '.csv'.")
                return self.SHAP_local
            else:
                if save_to_path is not None:
                    try:
                        if not os.path.exists(os.path.dirname(save_to_path)):
                            os.makedirs(os.path.dirname(save_to_path))
                        self.SHAP_global.to_csv(save_to_path)
                    except:
                        raise ValueError(
                            "The specified path does not work. The path should end with '.csv'.")
                return self.SHAP_global
        except:
            raise Exception(
                "To obtain SHAP values, refit with loss_based=False.")

    def get_SAGE_values(self,
                        save_to_path: Optional[str] = None) -> Union[pd.Series, pd.DataFrame]:
        """
        Returns estimates for global SAGE values from the fit()-function.
        SAGE values estimate each feature's contribution to the test set performance and hence are a loss-based feature importance measure.
        Negative SAGE values indicate the degree to which features are enhancing the model performance while positive SAGE values indicate that features may be detrimental to the model performance.

        Args:
            save_to_path (Optional[str], optional): If provided, save the dataframe to the specified path (relative to /consulting_tabpfn). Should end with '.csv'. Defaults to None.

        Raises:
            Exception: If the specified path to save the results does not work.
            Exception: If fit() has been applied with loss_based=False instead of loss_based=True or not at all.

        Returns:
            Union[pd.Series, pd.DataFrame]: Dataframe or Series of SAGE value per feature with the corresponding intercept of the regression model (corresponds to performance for the average prediction with empty feature coalition).
        """
        try:
            if save_to_path is not None:
                try:
                    if not os.path.exists(os.path.dirname(save_to_path)):
                        os.makedirs(os.path.dirname(save_to_path))
                    self.SAGE_global.to_csv(save_to_path)
                except:
                    raise ValueError(
                        "The specified path does not work. The path should end with '.csv'.")
            return self.SAGE_global
        except:
            raise Exception(
                "To obtain SAGE values, refit with loss_based=True.")

    def plot_bar(self, loss_based=False):
        """
        Plots global SHAP or SAGE values as a barplot. x-axis corresponds to features and the y-axis describes the associated global SHAP or SAGE values.

        Args:
            loss_based (bool, optional): Whether to plot SAGE values (explaining model performance) instead of SHAP values. Defaults to False.

        Raises:
            Exception:  If the queried configuration has not been fit.
        """

        try:
            if not loss_based:
                df = self.SHAP_global
                y_label = "SHAP"
                plot_title = "Global SHAP value per feature"
                palette = sns.color_palette(
                    "ch:s=.25,rot=-.25", n_colors=df.shape[1])
            else:
                df = self.SAGE_global
                y_label = "SAGE"
                plot_title = "SAGE value per feature"
                palette = sns.diverging_palette(10, 220, sep=3, n=df.shape[1])

            df = df[df.sum().sort_values(ascending=False).index]

            ax = sns.barplot(data=df, palette=palette)
            plt.title(plot_title)
            plt.xlabel("Feature")
            plt.ylabel(y_label)
            plt.xticks(rotation=90)
            plt.gcf().set_size_inches(8, 6)  # Adjust the width and height as needed
            # Adjust the bottom margin so feature-labels are not cut off
            plt.subplots_adjust(bottom=0.25)

            plt.show()

        except:
            raise Exception(
                "Barplot is only possible if fit() has been applied in advance.")

    def get_SAGE_t_test(self,
                        save_to_path: Optional[str] = None) -> Dict[str, Tuple[float, float]]:
        """
        One-tailed t-test to test for each feature whether it is detrimental for the model performance using SAGE values.
        H0: beta_j <= 0, H1: beta_j > 0 for j = 1, ..., p
        (Under H1 the loss increases due to the feature and hence it is detrimental for model performance) 
        Test statistic t_j= (beta_hat_j) / (se_j) where se_j= Var(beta_hat_j)^(1/2)_hat
        t_j is t-distributed with (n-p) df
        Reject H0 if t_j > t_(1-alpha/p)(n-p) (Bonferroni correction: Ensures that FWER <= alpha [Prob. of one or more type 1 errors in multiple testing])

        Source: Fahrmeir Regression S. 131

        Args:
            save_to_path (Optional[str], optional): If provided, save the dataframe to the specified path. The path should end with '.pkl'. Defaults to None.

        Raises:
            Exception: If fit() has been applied with loss_based=False instead of loss_based=True or not at all.

        Returns:
            Dict[int, Tuple[float, float]]: Dictionary with feature names as keys and tuple of corresponding coefficients and p-values as value for all features that are statistically significant at the alpha level 0.05.
        """
        if self.loss_based:
            results_dict = {}
            for param in list(self.lm_res.params.index):
                if param != "const":  # Ignore intercept
                    alpha = 0.05
                    # Bonferroni correction
                    corrected_alpha = alpha / (len(self.lm_res.params)-1)
                    coef = self.lm_res.params[param]
                    se = self.lm_res.bse[param]
                    t_stat = coef / se
                    p_val = 1 - stats.t.cdf(t_stat, df=self.lm_res.df_resid)
                    if p_val < corrected_alpha:
                        results_dict[self.data.feature_names[int(param)]] = {
                            "Coefficient": coef, "p_value": p_val}
            if save_to_path:
                try:
                    if not os.path.exists(os.path.dirname(save_to_path)):
                        os.makedirs(os.path.dirname(save_to_path))
                    with open(save_to_path, 'wb') as f:
                        pickle.dump(results_dict, f)
                except:
                    raise ValueError(
                        "The specified path does not work. The path should end with '.pkl'.")
            return results_dict
        else:
            raise Exception(
                "t-test is only possible with SAGE values. Refit with loss_based=True.")
