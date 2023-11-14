import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Union, List, Optional, Dict
from datetime import datetime
import time
from tabpfniml.methods.interpret import TabPFN_Interpret
from tabpfniml.datasets.datasets import dataset_iml

class ALE(TabPFN_Interpret):
    """ 
    Implementation of accumulated local effects (ALE). 
    ALE uses the idea of integrating partial derivatives to extract feature effects. 
    Compared to ICE/PD plots, it does not suffer from issues caused by extrapolating.
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
            data (dataset_iml, optional): The dataset that TabPFN's behavior shall be explained for.
            n_train (int, optional): The amount of train-samples to fit TabPFN on. Should not be larger than 1024. Defaults to 1024.
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
            features: Union[None, List[int], List[str]] = None,
            max_intervals_per_feature: int = 10,
            center: bool = True,
            discretize_by_linear_spacing=False):
        """
        Fit accumulated local effects (ALE) for several features. 
        For each feature, the feature space is discretized into K intervals and partial derivatives w.r.t this feature are approximated 
        by prediction differences of observations within this interval for the lower and upper interval bounds keeping all other features
        constant (local effects). Accumulating the local effects over intervals leads to the ALE-values.

        It has to be noted that ALE can only be fitted for continuous values.

        Args:
            features (Union[None, List[int], List[str]], optional): If specified, a list of indices or names of continuous features to fit the ALE. Since ALE cannot be fitted for categorical features, they cannot be passed. If not provided, ALE are estimated for all continuous features. Defaults to None.
            max_intervals_per_feature (int, optional): The maximum number of intervals to use when discretizing features. Ignored if a feature has fewer levels. Defaults to 10.
            center (bool, optional): Whether the ALE estimate should be centered so that the mean effect is zero. Defaults to True.
            discretize_by_linear_spacing (bool, optional): Whether to discretize features into K intervals using linear spacing instead of quantiles estimated on the train set. Defaults to False.

        Raises:
            Exception: If features have been specified, however in a non-valid format or it contains names or indices of features that do not exist or are categorical.
        """
        # TODO: Provide min_max_intervals manually.

        self.max_intervals_per_feature = max_intervals_per_feature
        self.discretize_by_linear_spacing = discretize_by_linear_spacing

        # If features were specified, map them to a list of feature indices (if possible).
        if features is not None:
            is_all_strings = all(isinstance(item, str) for item in features)
            is_all_ints = all(isinstance(item, int) for item in features)

            if is_all_strings:
                try:
                    features = [self.data.feature_name_to_index(
                        feature_name) for feature_name in features]
                except:
                    raise Exception(
                        "A feature has been passed as a string that does not occur in the dataset.")
            elif is_all_ints:
                for feature_index in features:
                    assert (
                        (feature_index+1) < self.data.num_features), "Feature {} does not occur in the dataset".format(str(feature_index))
            else:
                raise Exception(
                    "The features have to be specified either as a valid list of strings (of feature names) or as a valid list of ints (of feature indices).")

            categorical_features_passed = set(features) & set(
                self.data.categorical_features_idx)
            if len(categorical_features_passed) > 0:
                raise Exception("Feature(s) {} are categorical, ALE cannot be computed for categorical features.".format(
                    list(np.array(self.data.feature_names)[list(categorical_features_passed)])))

        # Otherwise take all continuous features
        else:
            features = self.data.continuous_features_idx

        self.features_idx = features

        self.classifier.fit(self.X_train, self.y_train)

        # If fit is called for the first time initialize relevant objects
        if not hasattr(self, "initialized"):
            # The number of observations to compute local effects for each feature and interval.
            self.n_obs_in_intervals = {}
            # Dictionary storing the interval bounds and corresponding ALE values for each feature.
            self.features_ale = {}
            # Dictionary storing whether ALE for feature is centered.
            self.centered = {}
            self.initialized = True

        elif self.initialized:
            print("ALE has already been fitted for features {}.".format(
                list(self.features_ale.keys())))
            features_to_refit = set(self.features_ale.keys()).intersection(
                self.data.feature_names[self.features_idx])
            if len(features_to_refit) > 0:
                print("Refitting ALE curves for features {}.".format(
                    list(features_to_refit)))

        for feature_idx in self.features_idx:
            feature_name = self.data.feature_names[feature_idx]

            local_effects = []
            obs_in_intervals = []
            n_obs_in_intervals = []

            if discretize_by_linear_spacing:
                # Intervals evenly spaced over feature range
                feature_interval_bounds = np.linspace(self.data.min_vals[feature_idx],
                                                      self.data.max_vals[feature_idx],
                                                      max(2, min(self.data.levels_per_feature[feature_idx], max_intervals_per_feature+1)))
            else:
                # Feature range discretized by quantiles
                _, feature_interval_bounds = pd.qcut(self.X_test[:, feature_idx],  # TODO: Check whether to compute by train or test intervals
                                                     q=min(
                                                         self.data.levels_per_feature[feature_idx], max_intervals_per_feature),
                                                     labels=False,
                                                     retbins=True,
                                                     duplicates="drop")

                # If there is only one level, set this as lower and upper bound (Edge case)
                if len(feature_interval_bounds) == 1:
                    feature_interval_bounds[1] = feature_interval_bounds[0]

            obs_in_intervals.append(np.logical_and(self.X_test[:, feature_idx] >= feature_interval_bounds[0],
                                                   self.X_test[:, feature_idx] <= feature_interval_bounds[1]))
            n_obs_in_intervals.append(np.sum(obs_in_intervals[0]))
            X_test_intervals = np.concatenate([self.X_test[obs_in_intervals[0], :],
                                               self.X_test[obs_in_intervals[0], :]], axis=0)
            X_test_intervals[:n_obs_in_intervals[0],
                             feature_idx] = feature_interval_bounds[0]
            X_test_intervals[n_obs_in_intervals[0]:,
                             feature_idx] = feature_interval_bounds[1]

            for i in range(1, len(feature_interval_bounds)-1):
                # For each interval select observations with feature-value in the interval
                obs_in_intervals.append(np.logical_and(self.X_test[:, feature_idx] > feature_interval_bounds[i],
                                                       self.X_test[:, feature_idx] <= feature_interval_bounds[i+1]))
                # Count the amount of observations
                n_obs_in_intervals.append(np.sum(obs_in_intervals[i]))

                # Concat observations substituting feature values once with lower- and once with upper-bound of feature interval
                X_test_interval = np.concatenate([self.X_test[obs_in_intervals[i], :],
                                                  self.X_test[obs_in_intervals[i], :]], axis=0)
                X_test_interval[:n_obs_in_intervals[i],
                                feature_idx] = feature_interval_bounds[i]
                X_test_interval[n_obs_in_intervals[i]:,
                                feature_idx] = feature_interval_bounds[i+1]
                X_test_intervals = np.concatenate(
                    [X_test_intervals, X_test_interval], axis=0)

            # TODO: Not all test observations are considered, since some obs in test set are outside bounds of train set

            # Get predictions for lower- and upper-bound
            p_eval_one = self.classifier.predict_proba(X_test_intervals)[:, 1]

            # Compute local effects as prediction differences of observations within each interval
            # (for the lower and upper interval bounds keeping all other features constant)
            for n_obs_in_interval in n_obs_in_intervals:
                if n_obs_in_interval > 0:
                    local_effects.append(np.mean(p_eval_one[n_obs_in_interval:(
                        2*n_obs_in_interval)])-np.mean(p_eval_one[:n_obs_in_interval]))
                else:
                    local_effects.append(0)
                p_eval_one = np.delete(p_eval_one, range(2*n_obs_in_interval))

            self.n_obs_in_intervals[feature_name] = n_obs_in_intervals

            # Accumulate local effects to ALE values
            self.features_ale[feature_name] = pd.DataFrame(np.concatenate([np.array([0]), np.cumsum(local_effects)]),
                                                           index=feature_interval_bounds,
                                                           columns=["ALE"])

            # If specified, center so that the mean effect is zero
            # TODO: Can we do this in get_ale such that all self.features_ale are consistent?
            if center:
                mean_ale_effect = np.sum(np.squeeze(np.array(self.features_ale[feature_name])[1:]) * np.array(
                    self.n_obs_in_intervals[feature_name])) / np.sum(self.n_obs_in_intervals[feature_name])
                self.features_ale[feature_name] = self.features_ale[feature_name] - \
                    mean_ale_effect
                self.centered[feature_name] = True
            else:
                self.centered[feature_name] = False

    def get_fitted_features(self) -> List:
        """
        Returns the names of all features that have previously been fitted.

        Returns:
            List: Names of fitted features.
        """
        return list(self.features_ale.keys())

    def get_ALE(self,
                feature: Union[str, int],
                save_to_path: Optional[str] = None) -> pd.DataFrame:
        """
        Return estimates of ALE for a feature together with the bounds and number of observations per interval.
        ALE increasing over intervals indicate that increasing the feature value has a positive impact on predictions, 
        while the contrary suggests a negative impact.

        Args:
            feature (Union[str, int]): The name or index of the feature whose ALE shall be returned. The feature must have been fitted previously.
            save_to_path (Optional[str], optional): If provided, save the dataframe to the specified path. Should end with '.csv'. Defaults to None.

        Raises:
            Exception: If the specified path to save the results does not work.
            Exception:  If the queried feature does not occur in the dataset, has not been specified correctly or has not been fited. To check for which features ALE were fitted, use get_fitted_features() method.

        Returns:
            pd.DataFrame: A pandas dataframe with the ALE and amount of observations per interval for a feature.
        """
        is_string = isinstance(feature, str)
        is_int = isinstance(feature, int)

        if is_string:
            assert (feature in self.data.feature_names), "Feature {} does not occur in the dataset".format(
                str(feature))
        elif is_int:
            try:
                feature = self.data.feature_names[feature]
            except:
                raise Exception(
                    "Feature {} does not occur in the dataset".format(str(feature)))
        else:
            raise Exception(
                "The feature has to be specified either as a valid string or as a valid int.")

        feature_name = feature

        assert feature_name in list(self.features_ale.keys(
        )), "The feature {} has not been fitted yet.".format(feature_name)

        intervals_str = ["({:.3f},{:.3f})".format(lower_bound, upper_bound) for lower_bound, upper_bound in zip(
            self.features_ale[feature_name].index[:-1], self.features_ale[feature_name].index[1:])]
        ale_df = pd.DataFrame(pd.concat([pd.Series(intervals_str),
                                        pd.Series(self.features_ale[feature_name].values.squeeze()[1:])],
                                        axis=1, ignore_index=True))
        ale_df.columns = ["Interval Bounds", "ALE"]

        if save_to_path is not None:
            try:
                if not os.path.exists(os.path.dirname(save_to_path)):
                    os.makedirs(os.path.dirname(save_to_path))
                ale_df.to_csv(save_to_path)
            except:
                raise ValueError(
                    "The specified path does not work. The path should end with '.csv'.")
        return ale_df.round(3)

    def plot(self,
             features: Union[None, List[int], List[str]] = None,
             save_to_dir: Optional[str] = None,
             file_names_save: Optional[Dict] = None):
        """
        Plots ALE for a list of features and allows storing the plots in a provided path.
        On the x-axis the feature values are shown and on the y-axis the ALE.
        An increasing curve indicates that increasing the feature value has a positive impact on predictions, 
        while a decreasing curve suggests a negative impact.

        Args:
            features (Union[None, List[int], List[str]], optional): If specified, a list of indices or names of features whose ALE shall be plotted. The features must have been fitted previously. Defaults to 0.
            save_to_dir (Optional[str], optional): If provided, save the plots to the specified directory. Defaultss to None.
            file_names_save (Dict, optional): Dictionary of file names to save plot. If not provided, will save plots as ale_ + time stamp. Should end with '.png'. Defaults to None.

        Raises:
            Exception: If the queried features does not occur in the dataset, has not been specified correctly or has not been fited. To check for which features ALE were fitted, use get_fitted_features() method.
        """
        if features is not None:
            is_all_strings = all(isinstance(item, str) for item in features)
            is_all_ints = all(isinstance(item, int) for item in features)

            if is_all_strings:
                try:
                    [self.data.feature_name_to_index(
                        feature_name) for feature_name in features]
                except:
                    raise Exception(
                        "A feature has been passed as a string that does not occur in the dataset.")
            elif is_all_ints:
                for feature_index in features:
                    assert (
                        (feature_index + 1) < self.data.num_features), "Feature {} does not occur in the dataset".format(str(feature_index))
                features = list(np.array(self.data.feature_names)[features])
            else:
                raise Exception(
                    "The features have to be specified either as a valid list of strings (of feature names) or as a valid list of ints (of feature indices).")
        else:
            features = list(self.features_ale.keys())

        feature_names = features

        for j, feature_name in enumerate(feature_names):
            assert feature_name in list(self.features_ale.keys(
            )), "The feature {} has not been fitted yet.".format(feature_name)

            plt.figure(j)
            sns.lineplot(data=self.features_ale[feature_name],
                         legend=False)

            sns.rugplot(x=pd.Series(self.X_test[:, self.data.feature_name_to_index(feature_name)]),
                        color="black")

            plt.xlabel("Feature values of \"{}\"".format(self.data.feature_complete_names[feature_name] if feature_name in list(
                self.data.feature_complete_names.keys()) else feature_name))
            plt.ylabel(
                "Centered ALE" if self.centered[feature_name] else "ALE")
            plt.title("ALE plot for target \"{}\"".format(
                self.data.id_to_target_name[self.data.id]))
            # TODO: Subset of x-axis such that plot is not too skewed by outliers.

            if save_to_dir:
                if not os.path.exists(save_to_dir):
                    os.makedirs(save_to_dir)
                if file_names_save:
                    try:
                        plt.savefig(
                            save_to_dir + "/" + file_names_save[feature_name] + ".pdf", format="pdf")
                    except:
                        Warning(
                            "No or invalid file name was provided for {}. Will save as ice_pd_ + time stamp.".format(feature_names))
                        plt.savefig(
                            save_to_dir + "/ale_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".pdf", format="pdf")
                        time.sleep(1)
                else:
                    plt.savefig(
                        save_to_dir + "/ale_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".pdf", format="pdf")
                    time.sleep(1)
