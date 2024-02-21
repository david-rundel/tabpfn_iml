from abc import ABC, abstractmethod
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tabpfniml.tabpfn_interpret import TabPFNClassifier
from tabpfniml.datasets.datasets import dataset_iml
from typing import Optional, List
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from netcal.metrics import ECE, MCE

class TabPFN_Interpret(ABC):
    """
    Abstract parent-class for all TabPFN-interpretability methods.
    All sub-classes implement the fit() function and various further functions (e.g. to return, save or plot the results of fit()), depending on the method.
    """

    def __init__(self,
                 data: dataset_iml,
                 n_train: int = 1024,
                 n_test: int = 512,
                 N_ensemble_configurations: int = 16,
                 store_gradients: bool = False,
                 device: str = "cpu",
                 debug: bool = False,
                 standardize_features: bool = False,
                 to_torch_tensor: bool = False):
        """
        Initialize a TabPFN-interpretability method by passing rudimentary objects and the general configuration that is constant across all flavors of the interpretability-method.
        For some variables it may be permitted to overwrite them in fit()-methods, although this is not intended in general.

        Args:
            data (dataset_iml, optional): The dataset that TabPFN's behavior shall be explained for. Defaults to dataset_iml.
            n_train (int, optional): The amount of train-samples to fit TabPFN on. Should not be larger than 1024. Defaults to 512.
            n_test (int, optional): The amount of test-samples to get predictions for. Defaults to 512.
            N_ensemble_configurations (int, optional): The amount of TabPFN forward passes with different augmentations ensembled. Defaults to 16.
            store_gradients (bool, optional): Modify the TabPFN model so that gradients in a backward-passes can be stored. Defaults to False.
            device (str, optional): The device to store tensors and the TabPFN model on. Defaults to "cpu".
            debug (bool, optional): Whether debug mode is activated. This leads to e.g. less train and test samples and can hence tremendously reduce computational cost. Overwrites various other parameters. Defaults to False.
            standardize_features (bool, optional): Whether to standardize each feature. Defaults to False.
            to_torch_tensor (bool, optional): Whether the data should be loaded as torch.tensor instead of np.ndarray.
        """

        self.data = data
        self.n_train = n_train
        self.n_test = n_test
        self.N_ensemble_configurations = N_ensemble_configurations
        self.store_gradients = store_gradients
        self.device = device
        self.debug = debug
        self.standardize_features = standardize_features
        self.to_torch_tensor = to_torch_tensor

        if self.debug:
            # Overwrite specification to reduce computational cost.
            self.n_train = 32
            self.n_test = 32
            self.N_ensemble_configurations = 2

        self.X_train, self.X_test, self.y_train, self.y_test = self.data.load(n_train=self.n_train,
                                                                              n_test=self.n_test,
                                                                              standardize_features_bool=self.standardize_features,
                                                                              to_torch_tensor=self.to_torch_tensor)

        self.classifier = TabPFNClassifier(device=self.device,
                                           N_ensemble_configurations=self.N_ensemble_configurations,
                                           store_gradients=self.store_gradients,
                                           seed= 42)

        # Prepare plotting library
        sns.set_style("ticks")
        plt.rcParams['figure.dpi'] = 120

        np.random.seed(42)

    @abstractmethod
    def fit(self):
        """
        Fit the interpretability-method to the provided dataset and TabPFN.
        All interpretability-method-specific parameters are passed only here, so different flavours of the interpretability method do not require a new object.
        Usually, several flavours can be fit at once and the default parameters lead to the most common flavour.
        """
        pass

    def check_performance(self, obs_per_bin: int = 10) -> pd.DataFrame:
        """
        Check performance of current train and test set w.r.t ROC AUC, accuracy, balanced accuracy,
        F1-score, precision, recall, and prediction time in seconds.

        Args:
            obs_per_bin (int, optional): Number of observations in each bin for ECE and MCE. Defaults to 10.
        
        Returns:
            pd.DataFrame: Dataframe of performance scores.
        """
        start = time.time()
        self.classifier.fit(self.X_train, self.y_train)
        p_eval = self.classifier.predict_proba(self.X_test)
        y_eval = self.classifier.classes_.take(
            np.asarray(np.argmax(p_eval, axis=-1), dtype=np.intp))
        prediction_time = time.time() - start

        df_performance = pd.DataFrame(columns=["Score"],
            index=["ROC AUC", "Accuracy", "Balanced accuracy", "F1-score",
                  "Precision", "Recall", "ECE", "MCE", "Prediction time in seconds"])
        
        df_performance.loc["ROC AUC", "Score"] = roc_auc_score(self.y_test, p_eval[:, 1])
        df_performance.loc["Accuracy", "Score"] = accuracy_score(self.y_test, y_eval)
        df_performance.loc["Balanced accuracy", "Score"] = balanced_accuracy_score(self.y_test, y_eval)
        df_performance.loc["F1-score", "Score"] = f1_score(self.y_test, y_eval)
        df_performance.loc["Precision", "Score"] = precision_score(self.y_test, y_eval)
        df_performance.loc["Recall", "Score"] = recall_score(self.y_test, y_eval)
        df_performance.loc["ECE", "Score"] = ECE(round(len(self.y_test)/obs_per_bin)).measure(p_eval[:,1], self.y_test)
        df_performance.loc["MCE", "Score"] = MCE(round(len(self.y_test)/obs_per_bin)).measure(p_eval[:,1], self.y_test)
        df_performance.loc["Prediction time in seconds", "Score"] = prediction_time

        return df_performance.astype(float).round(3)

    def check_ensemble_configurations_size(self,
                                           metric: str = "roc_auc",
                                           obs_per_bin: int = 10,
                                           ensemble_configuration_sizes: Optional[List] = None):
        """
        Check the performance of different ensemble configuration sizes w.r.t. to a certain metric.
        Allows to choose a the ensemble configuration size with sufficient performance to reduce computational effort for the interpretation methods.
        This method plots both the performance and the prediction time for different ensemble configurations.

        Args:
            metric (str, optional): Metric used to choose optimal ensemble configuration size. Defaults to "roc_auc". Options: "roc_auc", "accuracy", "balanced_accuracy", "f1_score", "precision", "recall"
            obs_per_bin (int, optional): Number of observations in each bin for ECE and MCE. Defaults to 10.
            ensemble_configuration_sizes (List, optional): List of number of ensemble configuration sizes to check performance. Defaults to None.
        """

        if hasattr(self, "metric_used"):
            if self.metric_used != metric:
                Warning("Metric {} does not correspond to previous used metric {}".format(
                    self.metric_used, metric))
        self.metric_used = metric

        metrics = {"roc_auc": "ROC AUC score", "accuracy": "Accuracy score", "balanced_accuracy": "Balanced accuracy score",
                   "f1_score": "F1 score", "precision": "Precision score", "recall": "Recall score",
                   "ece": "Expected calibration error", "mce": "Maximum calibration error"}
        assert metric in list(metrics.keys()), "Performance metric must be in {}".format(
            list(metrics.keys()))

        if not ensemble_configuration_sizes:
            ensemble_configuration_sizes = [2, 4, 8, 16, 32, 64, 128]

        col_name_performance = "Performance (" + metrics[metric] + ")"
        results = pd.DataFrame(index=ensemble_configuration_sizes, columns=[
                               col_name_performance, "Time"])
        results.index.set_names("Ensemble size", inplace=True)

        for ensemble_configurations_size in ensemble_configuration_sizes:

            start = time.time()
            classifier = TabPFNClassifier(device=self.device,
                                          N_ensemble_configurations=ensemble_configurations_size,
                                          store_gradients=False)

            classifier.fit(self.X_train, self.y_train)
            p_eval = classifier.predict_proba(self.X_test)
            y_eval = classifier.classes_.take(np.asarray(
                np.argmax(p_eval, axis=-1), dtype=np.intp))
            results.loc[ensemble_configurations_size, "Time"] = time.time() - start

            if metric == "roc_auc":
                results.loc[ensemble_configurations_size, col_name_performance] = roc_auc_score(
                    self.y_test, p_eval[:, 1])
            elif metric == "accuracy":
                results.loc[ensemble_configurations_size, col_name_performance] = accuracy_score(
                    self.y_test, y_eval)
            elif metric == "balanced_accuracy":
                results.loc[ensemble_configurations_size, col_name_performance] = balanced_accuracy_score(
                    self.y_test, y_eval)
            elif metric == "f1_score":
                results.loc[ensemble_configurations_size, col_name_performance] = f1_score(
                    self.y_test, y_eval)
            elif metric == "precision":
                results.loc[ensemble_configurations_size, col_name_performance] = precision_score(
                    self.y_test, y_eval)
            elif metric == "recall":
                results.loc[ensemble_configurations_size, col_name_performance] = recall_score(
                    self.y_test, y_eval)
            elif metric == "ece":
                results.loc[ensemble_configurations_size, col_name_performance] = ECE(
                    round(len(self.y_test)/obs_per_bin)).measure(p_eval[:,1], self.y_test)
            elif metric == "mce":
                results.loc[ensemble_configurations_size, col_name_performance] = MCE(
                    round(len(self.y_test)/obs_per_bin)).measure(p_eval[:,1], self.y_test)

        plt.figure(1)
        sns.lineplot(data=results[col_name_performance])
        plt.ylabel(metrics[metric])
        plt.title("Performance for test set of size {}\n for different ensemble sizes".format(
            self.n_test))

        plt.figure(2)
        sns.lineplot(data=results["Time"])
        plt.ylabel("Prediction time in seconds")
        plt.title("Prediction time for test set of size {}\n for different ensemble sizes".format(
            self.n_test))

    def check_train_set_size(self,
                             metric: str = "roc_auc",
                             obs_per_bin: int = 10,
                             n_resample: int = 10,
                             n_train_steps: Optional[List] = None,
                             store_best: bool = False):
        """
        Check the performance of different train set sizes w.r.t. to a certain metric.
        Allows to choose a subset of the train set with sufficient performance to reduce computational effort for the interpretation methods.
        This method plots both the performance and the prediction time for different train set sizes.

        Args:
            metric (str, optional): Metric used to choose optimal subset. Defaults to "roc_auc". Options: "roc_auc", "accuracy", "balanced_accuracy", "f1_score", "precision", "recall", "ece", "mce"
            obs_per_bin (int, optional): Number of observations in each bin for ECE and MCE. Defaults to 10.
            n_resample (int, optional): Number of resamples for each train set size to check distribution of performance. Defaults to 10.
            n_train_steps (Optional[List], optional): List of number of train observations to check performance. Defaults to None.
            store_best (bool, optional): Whether to store the indices of the best performance for each checked train set size. Defaults to False.
        """

        if hasattr(self, "metric_used"):
            if self.metric_used != metric:
                Warning("Metric {} does not correspond to previous used metric {}".format(
                    self.metric_used, metric))
        self.metric_used = metric

        metrics = {"roc_auc": "ROC AUC score", "accuracy": "Accuracy score", "balanced_accuracy": "Balanced accuracy score",
                   "f1_score": "F1 score", "precision": "Precision score", "recall": "Recall score", 
                   "ece": "Expected calibration error", "mce": "Maximum calibration error"}
        assert metric in list(metrics.keys()), "Performance metric must be in {}".format(
            list(metrics.keys()))

        if not n_train_steps:
            n_train_steps = [10, 20, 50, 100, 200, 300,
                             400, 500, 600, 700, 800, 900, 1000]

        n_train_steps_used = [
            n_train_step for n_train_step in n_train_steps if n_train_step < self.n_train] + [self.n_train]

        results_performance = pd.DataFrame(
            index=range(n_resample), columns=n_train_steps_used)
        results_performance.index.set_names("Resample step", inplace=True)
        results_performance.columns.set_names("Train set size", inplace=True)

        results_time = pd.DataFrame(index=range(
            n_resample), columns=n_train_steps_used)
        results_time.index.set_names("Resample step", inplace=True)
        results_time.columns.set_names("Train set size", inplace=True)

        if store_best:
            self.best_indices = {}
            self.best_metrics = {}

        indices_with_target_0 = np.arange(self.X_train.shape[0])[
            self.y_train == 0]
        indices_with_target_1 = np.arange(self.X_train.shape[0])[
            self.y_train == 1]
        share_of_ones = np.mean(self.y_train == 1)

        for n_train_used in n_train_steps_used:
            metric_score = 0
            for resample in range(n_resample):
                start = time.time()
                indices_subset_0 = np.random.choice(indices_with_target_0, max(
                    round(n_train_used*(1-share_of_ones)), 1), replace=False)
                indices_subset_1 = np.random.choice(indices_with_target_1, min(
                    round(n_train_used*share_of_ones), n_train_used-1), replace=False)
                indices = np.concatenate([indices_subset_0, indices_subset_1])
                np.random.shuffle(indices)
                self.classifier.fit(
                    self.X_train[indices, :], self.y_train[indices])
                p_eval = self.classifier.predict_proba(self.X_test)
                y_eval = self.classifier.classes_.take(
                    np.asarray(np.argmax(p_eval, axis=-1), dtype=np.intp))
                results_time.loc[resample, n_train_used] = time.time() - start

                if metric == "roc_auc":
                    results_performance.loc[resample, n_train_used] = roc_auc_score(
                        self.y_test, p_eval[:, 1])
                elif metric == "accuracy":
                    results_performance.loc[resample, n_train_used] = accuracy_score(
                        self.y_test, y_eval)
                elif metric == "balanced_accuracy":
                    results_performance.loc[resample, n_train_used] = balanced_accuracy_score(
                        self.y_test, y_eval)
                elif metric == "f1_score":
                    results_performance.loc[resample, n_train_used] = f1_score(
                        self.y_test, y_eval)
                elif metric == "precision":
                    results_performance.loc[resample, n_train_used] = precision_score(
                        self.y_test, y_eval)
                elif metric == "recall":
                    results_performance.loc[resample, n_train_used] = recall_score(
                        self.y_test, y_eval)
                elif metric == "ece":
                    results_performance.loc[resample, n_train_used] = ECE(round(len(self.y_test)/obs_per_bin)).measure(p_eval[:,1], self.y_test)
                elif metric == "mce":
                    results_performance.loc[resample, n_train_used] = MCE(round(len(self.y_test)/obs_per_bin)).measure(p_eval[:,1], self.y_test)

                if results_performance.loc[resample, n_train_used] > metric_score and store_best:
                    self.best_indices[n_train_used] = indices
                    self.best_metrics[n_train_used] = results_performance.loc[resample, n_train_used]
                    metric_score = results_performance.loc[resample, n_train_used]

        plt.figure(1)
        sns.boxplot(data=results_performance, orient="v")
        plt.ylabel(metrics[metric])
        plt.title("Distribution of performance for test set of size {}\n over different sizes of sampled train sets".format(
            self.n_test))

        plt.figure(2)
        sns.boxplot(data=results_time, orient="v")
        plt.ylabel("Prediction time in seconds")
        plt.title("Prediction time for test set of size {}\n over different sizes of sampled train sets".format(
            self.n_test))

    def find_optimal_train_subset(self,
                                  metric: str = "roc_auc",
                                  obs_per_bin: int = 10,
                                  n_train: int = 100,
                                  n_attempts: int = 10,
                                  init_with_best_in_check: bool = False,
                                  reinit: bool = False):
        """
        Allows to find the optimal subset of the train observations that is optimal w.r.t. a certain metric.

        Args:
            metric (str, optional): Metric used to choose optimal subset. Defaults to "roc_auc". Options: "roc_auc", "accuracy", "balanced_accuracy", "f1_score", "precision", "recall", "ece", "mce"
            obs_per_bin (int, optional): Number of observations in each bin for ECE and MCE. Defaults to 10.
            n_train (int, optional): Number of train samples to subset original train set. Defaults to 100.
            n_attempts (int, optional): Number of random samples from the train observations to check whether performance improves. Defaults to 10.
            init_with_best_in_check (bool, optional): Whether to initialize with best subset from the check performance funtion. Defaults to False.
            reinit (bool, optional): Whether to reinitialize the metric score to zero. Defaults to False.

        Returns:
            pd.DataFrame: Dataframe of performance scores.
        """

        if init_with_best_in_check:
            assert hasattr(
                self, "best_indices"), "The best subsets are not stored."
            self.indices_subset = self.best_indices[n_train]
            self.metric_score = self.best_metrics[n_train]
        elif reinit:
            self.metric_score = 0
        else:
            pass

        for attempt in range(n_attempts):
            indices = np.random.choice(
                np.arange(self.X_train.shape[0]), n_train, replace=False)
            self.classifier.fit(
                self.X_train[indices, :], self.y_train[indices])
            p_eval = self.classifier.predict_proba(self.X_test)
            y_eval = self.classifier.classes_.take(
                np.asarray(np.argmax(p_eval, axis=-1), dtype=np.intp))

            if metric == "roc_auc":
                metric_score_tmp = roc_auc_score(self.y_test, p_eval[:, 1])
            elif metric == "accuracy":
                metric_score_tmp = accuracy_score(self.y_test, y_eval)
            elif metric == "balanced_accuracy":
                metric_score_tmp = balanced_accuracy_score(self.y_test, y_eval)
            elif metric == "f1_score":
                metric_score_tmp = f1_score(self.y_test, y_eval)
            elif metric == "precision":
                metric_score_tmp = precision_score(self.y_test, y_eval)
            elif metric == "recall":
                metric_score_tmp = recall_score(self.y_test, y_eval)
            elif metric == "ece":
                metric_score_tmp = ECE(round(len(self.y_test)/obs_per_bin)).measure(p_eval[:,1], self.y_test)
            elif metric == "mce":
                metric_score_tmp = MCE(round(len(self.y_test)/obs_per_bin)).measure(p_eval[:,1], self.y_test)

            if metric_score_tmp > self.metric_score:
                self.indices_subset = indices
                self.metric_score = metric_score_tmp

        start = time.time()
        self.classifier.fit(
            self.X_train[self.indices_subset, :], self.y_train[self.indices_subset])
        p_eval = self.classifier.predict_proba(self.X_test)
        y_eval = self.classifier.classes_.take(
            np.asarray(np.argmax(p_eval, axis=-1), dtype=np.intp))
        prediction_time = time.time() - start

        df_performance = pd.DataFrame(columns=["Score"],
            index=["ROC AUC", "Accuracy", "Balanced accuracy", "F1-score",
                  "Precision", "Recall", "ECE", "MCE", "Prediction time in seconds"])
        
        df_performance.loc["ROC AUC", "Score"] = roc_auc_score(self.y_test, p_eval[:, 1])
        df_performance.loc["Accuracy", "Score"] = accuracy_score(self.y_test, y_eval)
        df_performance.loc["Balanced accuracy", "Score"] = balanced_accuracy_score(self.y_test, y_eval)
        df_performance.loc["F1-score", "Score"] = f1_score(self.y_test, y_eval)
        df_performance.loc["Precision", "Score"] = precision_score(self.y_test, y_eval)
        df_performance.loc["Recall", "Score"] = recall_score(self.y_test, y_eval)
        df_performance.loc["ECE", "Score"] = ECE(round(len(self.y_test)/obs_per_bin)).measure(p_eval[:,1], self.y_test)
        df_performance.loc["MCE", "Score"] = MCE(round(len(self.y_test)/obs_per_bin)).measure(p_eval[:,1], self.y_test)
        df_performance.loc["Prediction time in seconds", "Score"] = prediction_time

        return df_performance.astype(float).round(3)

    def subset_train_set(self, use_best: bool = True, n_train: int = 100):
        """
        Subset the train set and replace within object.

        Args:
            use_best (bool, optional): Whether to use the best subset w.r.t. a metric as determined in find_optimal_train_subset method. Defaults to True.
            n_train (int, optional): Number of train observations to subset. Defaults to 100.
        """

        if use_best:
            print("Subset train set with best subset of previous function.")
            self.X_train = self.X_train[self.indices_subset, :]
            self.y_train = self.y_train[self.indices_subset]
            self.n_train = len(self.indices_subset)
        else:
            print("Subset train set randomly.")
            indices = np.random.choice(
                np.arange(self.X_train.shape[0]), n_train, replace=False)
            self.X_train = self.X_train[indices, :]
            self.y_train = self.y_train[indices]
            self.n_train = len(indices)
