import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Dict
from datetime import datetime
import time
from tabpfniml.methods.interpret import TabPFN_Interpret
from tabpfniml.datasets.datasets import dataset_iml


class ICE_PD(TabPFN_Interpret):
    """ 
    Implementation of individual conditional expectations (ICE) curves and partial dependence (PD) plots. 
    ICE curves allow to uncover heterogeneity in the feature effects and PD plots indicate the expected target response over all test observations.
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
            max_levels_per_feature: int = 10):
        """
        Fit ICE curves of a chosen subset of features on a discretized grid for each observation in the test set.
        The PD plots are computed in the get function as average over all ICE curves.
        The ICE curves are stored in the object and can be amended by ICE curves for further features by calling the function again.
        If the ICE curves for certain features were already fitted, the previously fitted ICE curves will be overwritten.
        To clear the stored ICE curves, initialize the object again.

        Args:
            features (Union[None, List[int], List[str]], optional): If specified, a list of indices or names of features to fit the ICE curves. Defaults to None.
            max_levels_per_feature (int, optional): Maximum number of grid points to compute ICE curves. If the feature has less levels, the grid will only consist of the number of levels for that feature. Defaults to 10.

        Raises:
            Exception: If features have been specified, however in a non-valid format or it contains names or indices of features that do not exist or are categorical.
        """

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
                        (feature_index + 1) < self.data.num_features), "Feature {} does not occur in the dataset".format(str(feature_index))
            else:
                raise Exception(
                    "The features have to be specified either as a valid list of strings (of feature names) or as a valid list of ints (of feature indices).")
        else:
            features = list(np.arange(self.X_train.shape[1]))

        self.features_idx = features

        # TODO: Possible to only compute for subset?
        # min_vals = np.min(self.X_train, axis=0)
        # max_vals = np.max(self.X_train, axis=0)
        # levels_per_feature = np.array([len(np.unique(self.X_train[:,col])) for col in range(self.X_train.shape[1])])

        self.classifier.fit(self.X_train, self.y_train)

        if not hasattr(self, "initialized"):
            self.ice_curves = {}
            self.cat_features = {}
            self.initialized = True
        elif self.initialized:
            print("ICE and PD for features {} already fitted.".format(
                list(self.ice_curves.keys())))
            feature_fitted = set(self.ice_curves.keys()).intersection(
                self.data.feature_names[self.features_idx])
            if len(feature_fitted) > 0:
                print("Refitting ICE and PD curves for features {}.".format(
                    list(feature_fitted)))

        for feature_idx in self.features_idx:

            ice_curves_per_feat = []

            # TODO: Make sure that grid corresponds to grid for feature levels if less than max.
            if feature_idx in self.data.categorical_features_idx:
                feature_vals = np.sort(
                    np.unique(self.X_train[:, feature_idx])).astype(int)
                self.cat_features[self.data.feature_names[feature_idx]] = True
            else:
                if self.data.levels_per_feature[feature_idx] > max_levels_per_feature:
                    feature_vals = np.linspace(
                        self.data.min_vals[feature_idx], self.data.max_vals[feature_idx], max_levels_per_feature)
                else:
                    feature_vals = np.sort(
                        np.unique(self.X_train[:, feature_idx])).astype(int)
                self.cat_features[self.data.feature_names[feature_idx]] = False

            X_test_effects = []
            for test_idx in range(self.X_test.shape[0]):
                X_test_effects_obs = np.tile(
                    self.X_test[test_idx, :], (len(feature_vals), 1))
                X_test_effects_obs[:, feature_idx] = feature_vals
                X_test_effects.append(X_test_effects_obs)

            X_test_effects = np.concatenate(X_test_effects, axis=0)

            p_eval_one = self.classifier.predict_proba(X_test_effects)[:, 1]

            for test_idx in range(self.X_test.shape[0]):
                id_lower = len(feature_vals)*test_idx
                id_upper = len(feature_vals)*(test_idx+1)
                ice_curves_per_feat.append(p_eval_one[id_lower:id_upper])

            self.ice_curves[self.data.feature_names[feature_idx]] = pd.DataFrame(np.array(
                ice_curves_per_feat).transpose(), index=feature_vals, columns=np.arange(len(ice_curves_per_feat)))

    def get_fitted_features(self) -> List:
        """
        Returns the names of all features that have previously been fitted.

        Returns:
            List: Names of fitted features.
        """
        return list(self.features_ale.keys())

    def get_PD(self,
               feature: Union[str, int],
               save_to_path: Optional[str] = None) -> pd.DataFrame:
        """
        Return estimates of PD for a feature together with the grid.
        PD increasing indicate that increasing the feature value has a positive impact on predictions, 
        while the contrary suggests a negative impact.

        Args:
            feature (Union[str, int]): The name or index of the feature whose PD shall be returned. The feature must have been fitted previously.
            save_to_path (Optional[str], optional): If provided, save the dataframe to the specified path. Should end with '.csv'. Defaults to None.

        Raises:
            Exception: If the specified path to save the results does not work.
            Exception:  If the queried feature does not occur in the dataset, has not been specified correctly or has not been fited. To check for which features ALE were fitted, use get_fitted_features() method.

        Returns:
            pd.DataFrame: A pandas dataframe with the PD values and the grid values.
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

        assert feature_name in list(self.ice_curves.keys(
        )), "The feature {} has not been fitted yet.".format(feature_name)

        pd_df = pd.DataFrame(pd.concat([pd.Series(self.ice_curves[feature_name].index),
                                       pd.Series(self.ice_curves[feature_name].mean(axis=1).values.squeeze())],
                                       axis=1, ignore_index=True))
        pd_df.columns = ["Grid points", "PD"]

        if save_to_path is not None:
            try:
                if not os.path.exists(os.path.dirname(save_to_path)):
                    os.makedirs(os.path.dirname(save_to_path))
                pd_df.to_csv(save_to_path)
            except:
                raise ValueError(
                    "The specified path does not work. The path should end with '.csv'.")
        return pd_df.round(3)

    def plot(self,
             features: Union[List[int], List[str], None] = None,
             plot_classification_threshold: bool = True,
             center: bool = False,
             save_to_dir: Optional[str] = None,
             file_names_save: Optional[Dict] = None,
             feature_to_subset: Union[str, None] = None,
             subset_threshold: Union[float, None] = None,
             use_larger_than_theshold=False):
        """
        Plots ICE and PD plots for subset of features and allows to store plots in provided directory.
        Make sure that ICE curves have been fitted in the fit function. To check which ICE curves were fitted use check_fitted() method.
        Also allows to plot ICE and PD plots for subset of observations based on setting a threshold to subset for one feature.

        Args:
            features (Union[None, List[int], List[str]], optional): If specified, a list of indices or names of features to plot the ICE and PD plots. Defaults to None.
            plot_classification_threshold (bool, optional): Whether to plot the classification threshold as dotted line. Defaults to True.
            center (bool, optional): Whether ICE curve should be centered at minimum feature value.
            save_to_dir (str, optional): If provided, save the plots to the specified directory.
            file_names_save (Dict, optional): Dictionary of file names to save plot. If not provided, will save plots as ice_pd_ + time stamp. Defaults to None.
            feature_to_subset (Union[str, None], optional): Name of feature used to subset ICE curves and PD plot. Defaults to None.
            subset_threshold (Union[float, None], optional): Threshold used for subsetting. Defaults to None.
            use_larger_than_theshold (bool, optional): Whether the subset should be above or below threshold. Defaults to False.

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
                        (feature_index+1) < self.data.num_features), "Feature {} does not occur in the dataset".format(str(feature_index))
                features = list(np.array(self.data.feature_names)[features])
            else:
                raise Exception(
                    "The features have to be specified either as a valid list of strings (of feature names) or as a valid list of ints (of feature indices).")
        else:
            features = list(self.ice_curves.keys())

        feature_names = features

        # TODO Make sure that subset is the correct size of n_test
        if feature_to_subset is not None:
            if use_larger_than_theshold:
                subset_idx = np.arange(self.X_test.shape[0])[
                    self.X_test[:, self.data.feature_name_to_index(feature_to_subset)] > subset_threshold]
            else:
                subset_idx = np.arange(self.X_test.shape[0])[
                    self.X_test[:, self.data.feature_name_to_index(feature_to_subset)] <= subset_threshold]
        else:
            subset_idx = np.arange(self.X_test.shape[0])

        for j, feature_name in enumerate(feature_names):
            assert feature_name in list(self.ice_curves.keys(
            )), "Feature {} not fitted.".format(feature_name)

            plt.figure(j)

            if self.cat_features[feature_name]:
                if center:
                    Warning(
                        "ICE boxplots for categorical features cannot be centered, will use uncentered instead.")
                sns.boxplot(
                    data=self.ice_curves[feature_name].loc[:, subset_idx].transpose())
            else:
                if center:
                    ice_curves_used = self.ice_curves[feature_name].iloc[:,
                                                                         subset_idx] - self.ice_curves[feature_name].iloc[0, subset_idx]
                else:
                    ice_curves_used = self.ice_curves[feature_name].iloc[:, subset_idx]

                sns.lineplot(data=ice_curves_used, palette=sns.color_palette(
                    ['black'], len(subset_idx)), linewidth=0.15, legend=False)

                sns.lineplot(data=ice_curves_used.mean(
                    axis=1), palette="red", linewidth=2)

                sns.rugplot(x=pd.Series(self.X_test[subset_idx, self.data.feature_name_to_index(
                    feature_name)]), color="black")

            if plot_classification_threshold and not center:
                plt.axhline(y=0.5, color="blue", linestyle="dashed")
            if center:
                plt.axhline(y=0, color="blue", linestyle="dashed")

            plt.xlabel("Feature values of \"{}\"".format(self.data.feature_complete_names[feature_name] if feature_name in list(
                self.data.feature_complete_names.keys()) else feature_name))
            if center:
                plt.ylabel("Probability difference of \"{}\"".format(
                    self.data.id_to_target_name[self.data.id]))
            else:
                plt.ylabel("Probability of \"{}\"".format(
                    self.data.id_to_target_name[self.data.id]))
            if self.cat_features[feature_name]:
                plt.title("ICE and PD boxplots for target \"{}\"".format(
                    self.data.id_to_target_name[self.data.id]))
            else:
                plt.title("ICE and PD curves for target \"{}\"".format(
                    self.data.id_to_target_name[self.data.id]))

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
                        plt.savefig(save_to_dir + "/ice_pd_" + datetime.now().strftime(
                            "%Y-%m-%d-%H-%M-%S") + ".pdf", format="pdf")
                        time.sleep(1)
                else:
                    plt.savefig(save_to_dir + "/ice_pd_" + datetime.now().strftime(
                        "%Y-%m-%d-%H-%M-%S") + ".pdf", format="pdf")
                    time.sleep(1)
