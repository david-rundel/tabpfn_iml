import pandas as pd
import numpy as np

from tabpfniml.methods.interpret import TabPFN_Interpret
from tabpfniml.datasets.datasets import dataset_iml

import shap
from numpy.testing import assert_almost_equal
from typing import Optional, Union
import os


class SHAP_Package_Wrapper(TabPFN_Interpret):
    """
    Wrapper-class around the SHAP-package from Scott Lundberg (https://shap.readthedocs.io/en/latest/index.html) in order to make it
    utilizable with TabPFN.

    Kernel SHAP is a model agnostic method used for explaining model predictions. 
    It estimates the Shapley values, which assign each feature's contribution to the prediction based on its interaction with other features.
    In this implementation, SHAP values explain the difference from the model prediction when using the median per feature over training observations.
    Hence, all features in an observation, that are equal to the median per feature, have no feature effect by design.

    Compared to our implementation (iml/methods/kernel_shap.py) it offers a big variety of plots. However, it is less performant when using it 
    in combination with TabPFN. 

    In detail, SHAP approximates marginalizing the prediction function over non-coalition features of a test-sample by substituting them with values of the median sample.
    However, the TabPFN model can be refit with negible computational overhead leading to an exact retraining with only the coalition features.
    For this approach, TabPFN can only evaluate one feature coalition at once (because every feature coalition requires refitting the model on the training data) 
    but with several test-samples at the same time. In this implementation of SHAP however, it is first iterated over test samples (shap_values() => explain()) 
    and afterwards over all feature coalitions (run()). 
    In order to obtain exact retraining, the forward pass of TabPFN has to be called for each test-sample and coalition seperately. 
    This requires (n_test * K) forward passes with 1 test sample in each forward pass. This approach (sequenial inference) is generally less efficient than
    batch inference, since it does not leverage parallel processing capabilities (especially on GPUs).
    Alternatively, using the median values for non-coalition features yields a faster (n_test forward passes with K test samples in each forward pass, where the model is 
    not refit for every test sample) but approximate solution.
    Our implementation yields the exact solution with only K forward passes (with n_test test samples in each forward pass).

    Note: Plots can only be displayed properly when executing them in Jupyter-notebooks.
    """
    class KernelExplainerForTabPFN(shap.KernelExplainer):
        """
        Class that inherits from shap.KernelExplainer and solely overwrites the run()-method.
        The forward-pass is modified in order to take information describing the coalitions as arguments.
        """

        def run(self):
            """
            run()-method of shap.KernelExplainer overwritten in order to modify forward-pass to take information describing the coalitions as arguments.
            This way test-samples only with coalitions-features and without median values for non-coalition features can be restored to obtain exact retraining over non-coalition features.
            This method is called in the explain()-method (which is called in the shap_values()-method for every test-sample) )in shap/explainers/_kernel.py of the Kernel class.
            """
            num_to_run = self.nsamplesAdded * self.N - self.nsamplesRun * self.N
            data = self.synth_data[self.nsamplesRun *
                                   self.N:self.nsamplesAdded*self.N, :]
            if self.keep_index:
                index = self.synth_data_index[self.nsamplesRun *
                                              self.N:self.nsamplesAdded*self.N]
                index = pd.DataFrame(index, columns=[self.data.index_name])
                data = pd.DataFrame(data, columns=self.data.group_names)
                data = pd.concat([index, data], axis=1).set_index(
                    self.data.index_name)
                if self.keep_index_ordered:
                    data = data.sort_index()

            # EDIT
            # Calls TabPFN_Wrapper-instance, but with additional maskMatrix and varyingFeatureGroups attributes
            # Previously: modelOut = self.model.f(data)
            modelOut = self.model.f(
                data, mask_matrix=self.maskMatrix, varying_feature_groups=self.varyingFeatureGroups)
            # END EDIT

            if isinstance(modelOut, (pd.DataFrame, pd.Series)):
                modelOut = modelOut.values
            self.y[self.nsamplesRun * self.N:self.nsamplesAdded *
                   self.N, :] = np.reshape(modelOut, (num_to_run, self.D))

            # find the expected value of each output
            for i in range(self.nsamplesRun, self.nsamplesAdded):
                eyVal = np.zeros(self.D)
                for j in range(0, self.N):
                    eyVal += self.y[i * self.N + j, :] * self.data.weights[j]

                self.ey[i, :] = eyVal
                self.nsamplesRun += 1

    def __init__(self,
                 data: dataset_iml,
                 n_train: int = 1024,
                 n_test: int = 512,
                 N_ensemble_configurations: int = 16,
                 device: str = "cpu",
                 debug: bool = False):
        """
        UPDATE DOCSTRING IF PARENT CLASSES DOCSTRING HAS CHANGED

        Initialize a TabPFN-interpretability method by passing rudimentary objects and the general configuration that is constant across all flavors of the interpretability-method.
        For some variables it may be permitted to overwrite them in fit()-methods, although this is not intended in general.
        Additionally to the parent-classes __init__()-method, the visualition tool is initialized.

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
                         store_gradients=False,
                         standardize_features=False,
                         to_torch_tensor=False)

        shap.initjs()

    def f(self,
          X_test: np.array,
          mask_matrix: Optional[np.array] = None,
          varying_feature_groups: Optional[np.array] = None) -> np.array:
        """
        Wrapper around the TabPFN-classifier used in KernelExplainerForTabPFN (that inherits from shap.KernelExplainer).
        Besides the test data, it takes information as arguments that can be used to restore test-sample with coalition-features only and without median values for non-coalition features.
        This is used to obtain exact retraining over non-coalition features.
        This method is called by the run()-method of the KernelExplainerForTabPFN class.

        Args:
            X_test (np.array): K duplicates of a single test-sample to perform inference on (one for each coalition). In each coalition, the non-coalition features are replaced by their median value in the train-set.
            mask_matrix (Optional[np.array], optional): Matrix indicating the features in each of K coalitions. Defaults to None.
            varying_feature_groups (Optional[np.array], optional): The features of the test-sample that are different from their median value in the train-set. Feature effects are only estimated for them. Defaults to None.

        Returns:
            np.array: Array of K predicted probabilities for the desired class.
        """
        if self.approximate_marginalization or ((mask_matrix is None) and (varying_feature_groups is None)):
            # Marginalize over non-coalition features via median values (as in original implementation).
            self.classifier.fit(self.X_train, self.y_train)
            return self.classifier.predict_proba(X_test)[:, self.class_to_be_explained]

        else:
            # Refit TabPFN for each coalition using only the coalition-features.
            K = X_test.shape[0]
            preds = []
            feature_subset_wo_varying_feature_groups = np.setdiff1d(
                np.arange(self.data.num_features), varying_feature_groups)
            for i in range(K):
                feature_subset_i = np.union1d(varying_feature_groups[np.array(
                    mask_matrix[i], dtype=bool)], feature_subset_wo_varying_feature_groups)

                X_train_subset = self.X_train[:, feature_subset_i]
                y_train_subset = self.y_train
                X_test_subset = X_test[i, feature_subset_i].reshape(1, -1)

                self.classifier.fit(X_train_subset, y_train_subset)
                preds.append(self.classifier.predict_proba(
                    X_test_subset)[:, self.class_to_be_explained][0])
            return np.array(preds)

    def fit(self,
            K: int = 512,
            approximate_marginalization: bool = False,
            log_odds_units: bool = False,
            class_to_be_explained: int = 1,):
        """
        Estimates local SHAP values using the overwritten shap.KernelExplainer.

        Args:
            K (int, optional): Amount of coalitions to be sampled. Takes the role of amount of observations (n) in regression. To avoid n<p-problem ensure that it is greater than or equal to self.data.num_features. Enhance K to get more accurate estimates of local Shapley values. Defaults to 512.
            approximate_marginalization (bool, optional): Whether to apply the faster but approximate solution (that marginalizes over non-coalition features with median-values). Defaults to False.
            log_odds_units (bool, optional): Whether to use a logit-link to connect the SHAP values to the model output (SHAP values estimate effect in log-odds space instead of probability space, https://shap-lrjball.readthedocs.io/en/latest/example_notebooks/kernel_explainer/Squashing%20Effect.html). Defaults to True.
            class_to_be_explained (int, optional): The class that predictions are explained for. Defaults to 1.

        """
        if self.debug:
            # Must still be greater than or equal to self.data.num_features in order to avoid n<p-problem in OLS.
            self.K = 32
        else:
            self.K = K

        if log_odds_units:
            self.link_function = "logit"
        else:
            self.link_function = "identity"

        self.approximate_marginalization = approximate_marginalization
        self.class_to_be_explained = class_to_be_explained

        # Compute median per feature in training data
        med = np.expand_dims(a=np.median(self.X_train, axis=0),
                             axis=0)

        # Instantiate explainer
        self.explainer = self.KernelExplainerForTabPFN(self.f,
                                                       med,
                                                       link=self.link_function)

        self.SHAP_local = self.explainer.shap_values(self.X_test,
                                                     nsamples=self.K)  # nsamples corresponds to amount of coalitions

        if not log_odds_units:
            # Only possible for last test-sample, since self.explainer.fx is simply updated per test observation
            assert_almost_equal(
                self.SHAP_local[-1].sum(), (self.explainer.fx[0] - self.explainer.expected_value), decimal=3)

    def get_SHAP_values(self,
                        local: bool = False,
                        save_to_path: Optional[str] = None) -> pd.DataFrame:
        """
        Returns local or global SHAP values estimated in the fit()-function via explainer.shap_values().
        SHAP values estimate the Shapley values, which assign each feature's contribution to the prediction based on its interaction with other features.
        In this implementation, SHAP values explain the difference from the model prediction when using the median per feature over training observations.

        Positive local SHAP values indicate how much higher the prediction is due too the feature while negative local SHAP values inicate how much smaller it is.
        Global SHAP values are always positive and indicate the influence of a feature on the prediction. 

        Args:
            local (bool, optional): Whether to return local SHAP values (per test sample) or the absolute values averaged over test samples (global SHAP as feature importance measure). Defaults to False.
            save_to_path (Optional[str], optional): If provided, save the dataframe to the specified path. Should end with '.csv'.  Defaults to None.

        Raises:
            Exception: If the specified path does not work or fit() has not been executed yet.

        Returns:
            pd.DataFrame: Dataframe of local or global SHAP values where columns correspond to features and rows to test observations that are being explained (only one for global SHAP if local=False).
        """
        try:
            self.SHAP_local_df = pd.DataFrame(
                self.SHAP_local, columns=self.data.feature_names)

            if local:
                if save_to_path is not None:
                    try:
                        if not os.path.exists(os.path.dirname(save_to_path)):
                            os.makedirs(os.path.dirname(save_to_path))
                        self.SHAP_local_df.to_csv(save_to_path)
                    except:
                        raise ValueError(
                            "The specified path does not work. The path should end with '.csv'.")
                return self.SHAP_local_df
            else:
                self.SHAP_global_df = self.SHAP_local_df.abs().mean()
                if save_to_path is not None:
                    try:
                        if not os.path.exists(os.path.dirname(save_to_path)):
                            os.makedirs(os.path.dirname(save_to_path))
                        self.SHAP_global_df.to_csv(save_to_path)
                    except:
                        raise ValueError(
                            "The specified path does not work. The path should end with '.csv'.")
                return self.SHAP_global_df

        except:
            raise Exception(
                "Either the specified path does not work or fit() has to be executed first.")

    def plot_force(self,
                   test_index: Optional[int] = None):
        """
        Plot local SHAP values for a single or various test samples.
        The plot portrays, starting from the median prediction, how features have lead to a deviating prediction for one or many test-samples.

        If only one test sample shall be explained, the x-axis represents the prediction space.
        For multiple test samples, the x-axis represents the samples and the y-axis the prediction space.

        Args:
            test_index (Optional[int], optional): The index of the test sample to be explained. If no index is provided, the whole test set will be explained. Defaults to None.

        Raises:
            Exception: If the specified test_index is not appropriate or fit() has not been executed yet.
        """
        try:
            if test_index is not None:
                return shap.force_plot(base_value=self.explainer.expected_value,
                                       shap_values=self.SHAP_local[test_index],
                                       # .iloc
                                       features=self.X_test[test_index, :],
                                       link=self.link_function,
                                       feature_names=self.data.feature_names)

            else:
                return shap.force_plot(base_value=self.explainer.expected_value,
                                       shap_values=self.SHAP_local,
                                       features=self.X_test,
                                       link=self.link_function,
                                       feature_names=self.data.feature_names)
        except:
            raise Exception(
                "Either the specified test_index is not appropriate or fit() has to be executed first.")

    def plot_summary(self):
        """
        Plots a beeswarm plot of the local SHAP values.
        The plot indicates the distribution of SHAP values per feature together with the value of the feature per test sample.
        This function does not enable the use of a link function, e.g. to return the value on  log-odds units.

        Raises:
            Exception: If fit() has not been executed yet.
        """
        try:
            return shap.summary_plot(shap_values=self.SHAP_local,
                                     features=self.X_test,
                                     feature_names=self.data.feature_names)
        except:
            raise Exception("fit() has to be executed first.")

    def plot_dependence(self,
                        dependent_feature: Union[int, str] = 0):
        """
        Indicated how local SHAP values of a specific feature for test-samples depend on the corresponding feature values of the test-samples.
        This function does not enable the use of a link function, e.g. to return the value on  log-odds units.

        Args:
            dependent_feature (Union[int, str], optional): Index or name of a feature to be analyzed. Defaults to 0.

        Raises:
            Exception: If the specified dependent_feature is not appropriate or fit() has not been executed yet.
        """
        try:
            return shap.dependence_plot(ind=dependent_feature,
                                        shap_values=self.SHAP_local,
                                        features=self.X_test,
                                        feature_names=self.data.feature_names
                                        )
        except:
            raise Exception(
                "Either the specified dependent_feature is not appropriate or fit() has to be executed first.")
