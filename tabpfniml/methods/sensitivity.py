import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils import column_or_1d
import seaborn as sns
import matplotlib.pyplot as plt
from tabpfniml.methods.interpret import TabPFN_Interpret
from tabpfniml.datasets.datasets import dataset_iml
from typing import Union, Optional, List
import os


class Sensitivity(TabPFN_Interpret):
    """
    Implementation of Sensitivity-Analysis as gradient-based, model-specific interpretability method.
    In comparision to other model-classes, TabPFN allows the computation of gradients w.r.t train-samples.
    Therefore Sensitivity-Analysis is not only performed for features but also for train-samples.
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
        Compared to the parent-classes __init__()-method, it is ensured that the TabPFN model is modified for Sensitivity Analyis (E.g. gradients in backward-passes are stored)
        and that features are standardized (to ensure comparability of the gradient-intensity).

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
                         store_gradients=True,
                         standardize_features=True,
                         to_torch_tensor=False)

    def fit(self,
            compute_wrt_feature: bool = True,
            compute_wrt_observation: bool = False,
            loss_based: bool = False,
            pred_based: bool = True,
            class_to_be_explained: int = 1):
        """
        This method calculates the gradient of the model prediction (or alternatively loss) with 
        respect to the input features (or alternatively training observation). By doing so, 
        it determines the inputs that require minimal alteration while achieving the maximum 
        change in prediction (or alternatively loss). However, it only represents the sensitivity 
        of model outputs (or alternatively loss-values) to changes in the input while not 
        considering the extent to which the input may have already influenced it (no attribution-
        method).

        Modifies the TabPFN forward- and backward-pass to store gradients and infer 
        how sensitive the model predictions are to features or training observations.

        We have decided to implement this as a local method, meaning the the sensitivity scores are computed per
        test observation in isolation. They can afterwards be aggregated to global scores. Global scores could
        be estimated more efficiently, however then it would not be possible to obtain local scores.

        Args:
            compute_wrt_feature (bool, optional): Whether to estimate the effects/importance of features. Defaults to True.
            compute_wrt_observation (bool, optional): Wheter to compute the gradients for observations instead of features. Defaults to False.
            loss_based (bool, optional): Whether the model performance (instead of the model predictions) should be explained. Constitutes a Feature Importance instead of Feature Effects measure. Defaults to False.
            pred_based (bool, optional): Whether to estimate feature/observation effects. Defaults to True.
            class_to_be_explained (int, optional): The class that predictions are explained for. Ignored it loss_based=True. Defaults to 1.
        """

        def quality_checks(X_train, X_test, y_train):
            """
            Helper method
            Performs quality checks (from tabpfn script) before transforming numpy-array into tensor 
            (numpy-array not differentiable, hence in order to get gradients w.r.t obs we have cannot convert them to tensor)
            (Normally done in tabpfn/scripts/transformer_prediction_interface.py fit())
            """
            def _validate_targets(y):
                y_ = column_or_1d(y, warn=True)
                check_classification_targets(y)
                cls, y = np.unique(y_, return_inverse=True)
                if len(cls) < 2:
                    raise ValueError(
                        "The number of classes has to be greater than one; got %d class"
                        % len(cls)
                    )
                classes_ = cls
                return np.asarray(y, dtype=np.float64, order="C"), classes_

            X_train, y_train = check_X_y(
                X_train, y_train, force_all_finite=False)
            y_train, classes_ = _validate_targets(y_train)
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
            X_test = check_array(X_test, force_all_finite=False)

            return X_train, X_test, y_train, classes_

        self.compute_wrt_feature = compute_wrt_feature
        self.compute_wrt_observation = compute_wrt_observation
        self.loss_based = loss_based
        self.pred_based = pred_based
        self.class_to_be_explained = class_to_be_explained

        # Step 1: Quality Checks
        # Manually perform a range of quality checks on the data before transforming them from np.arrays into torch.tensors.
        # The quality checks are then left out in the TabPFN forward pass (Normally done in tabpfn/scripts/transformer_prediction_interface.py fit())
        # If done in the forward-pass, the quality checks would require the data to be in np.array-format.
        # This would interrupt the computational graph and stop the data from being a leaf node that gradients can be computed for.
        # Our approach ensuress that all quality-checks are conducted and we can compute gradients for the data.
        self.X_train, self.X_test, self.y_train, self.classifier.classes_ = quality_checks(X_train=self.X_train,
                                                                                           X_test=self.X_test,
                                                                                           y_train=self.y_train)

        # Map to torch.tensor as leaf node
        self.X_train = torch.tensor(
            self.X_train, requires_grad=True, device=self.device)
        self.X_test = torch.tensor(
            self.X_test, requires_grad=True, device=self.device)
        self.y_train = torch.tensor(self.y_train, device=self.device)
        self.y_test = torch.tensor(
            self.y_test, device=self.device).type(torch.LongTensor)

        # Define gradient hook to store gradients computed in the backward pass
        # https://discuss.pytorch.org/t/why-cant-i-see-grad-of-an-intermediate-variable/94/7
        temp_grads = {}

        def save_grad(name):
            def hook(grad):
                temp_grads[name] = grad
            return hook

        self.X_train.register_hook(save_grad("X_train"))
        self.X_test.register_hook(save_grad("X_test"))

        self.classifier.fit(self.X_train, self.y_train)
        preds = self.classifier.predict_proba(self.X_test)

        # Step 2: Set up dataframes
        if self.pred_based:
            if self.compute_wrt_feature:
                self.FE_local = pd.DataFrame()
            if self.compute_wrt_observation:
                self.OE_local = pd.DataFrame()

        if self.loss_based:
            # Tradiitonal LOCO as OI_local
            self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
            loss = self.criterion(preds, self.y_test)  # .detach().numpy()

            if self.compute_wrt_feature:
                self.FI_local = pd.DataFrame()
            if self.compute_wrt_observation:
                self.OI_local = pd.DataFrame()

        # Step 3: Compute sensitvity scores
        for i in range(self.n_test):
            if self.pred_based:
                preds[i, self.class_to_be_explained].backward(
                    retain_graph=True)

                if self.compute_wrt_feature:
                    # Sensitivity as Feature Effects (FE_local)
                    X_test_grads = torch.nan_to_num(
                        temp_grads["X_test"].squeeze()[i, :], nan=0)
                    temp_FE = pd.Series(X_test_grads.numpy()).abs()
                    self.FE_local = pd.concat([self.FE_local, pd.DataFrame([temp_FE])],
                                              ignore_index=True)

                if self.compute_wrt_observation:
                    # Sensitivity as Observation Effects (OE_local)
                    X_train_grads = torch.nan_to_num(
                        temp_grads["X_train"].squeeze(), nan=0)
                    sensitivity_x_train = torch.norm(X_train_grads, p=2, dim=1)
                    # Always positive due to norm.
                    temp_OE = pd.Series(sensitivity_x_train.numpy()).abs()
                    self.OE_local = pd.concat([self.OE_local, pd.DataFrame([temp_OE])],
                                              ignore_index=True)
                    # / temp_OE.sum() => Do not divide gradients by denominator, because otherwise every observation gets the same weight again
                    # But we want gradients per observation to have different weights depending on whether the classification was correct or not

            if self.loss_based:
                loss[i].backward(retain_graph=True)

                if self.compute_wrt_feature:
                    # Sensitivity as Feature Importance (FI_local)
                    X_test_grads = torch.nan_to_num(
                        temp_grads["X_test"].squeeze()[i, :], nan=0)
                    # Always positive since FI measure (intensity of effect on loss more important than direction)
                    temp_FI = pd.Series(X_test_grads.numpy()).abs()
                    self.FI_local = pd.concat([self.FI_local, pd.DataFrame([temp_FI])],
                                              ignore_index=True)

                if self.compute_wrt_observation:
                    # Sensitivity as Observation Importance (OI_local)
                    X_train_grads = torch.nan_to_num(
                        temp_grads["X_train"].squeeze(), nan=0)
                    # Always positive due to norm.
                    sensitivity_x_train = torch.norm(X_train_grads, p=2, dim=1)
                    temp_OI = pd.Series(sensitivity_x_train.numpy()).abs()
                    self.OI_local = pd.concat([self.OI_local, pd.DataFrame([temp_OI])],
                                              ignore_index=True)

        # Rename columns
        if self.pred_based:
            if self.compute_wrt_feature:
                self.FE_local.columns = [
                    "Sens_FE_" + feat_name for feat_name in self.data.feature_names]
            if self.compute_wrt_observation:
                self.OE_local.columns = ["Sens_OE_" +
                                         str(i) for i in range(self.n_train)]

        if self.loss_based:
            if self.compute_wrt_feature:
                self.FI_local.columns = [
                    "Sens_FI_" + feat_name for feat_name in self.data.feature_names]
            if self.compute_wrt_observation:
                self.OI_local.columns = ["Sens_OI_" +
                                         str(i) for i in range(self.n_train)]

        # Step 4: Obtain global measures
        if self.pred_based:
            if self.compute_wrt_feature:
                self.FE_global = self.FE_local.mean(axis=0)
            if self.compute_wrt_observation:
                self.OE_global = self.OE_local.mean(axis=0)

        if self.loss_based:
            if self.compute_wrt_feature:
                self.FI_global = self.FI_local.mean(axis=0)
            if self.compute_wrt_observation:
                self.OI_global = self.OI_local.mean(axis=0)

    def get_FI(self,
               local: bool = False,
               save_to_path: Optional[str] = None) -> Union[pd.Series, pd.DataFrame]:
        """
        Return estimates of Sensitivity FI values from the fit()-function.
        Sensitivity FI values esimate the sensitivity of the predictive performace on features. 
        Sensitivity FI values are always positive.

        Args:
            local (bool, optional): Whether to average the results across test-samples. Defaults to False.
            save_to_path (Optional[str], optional): If provided, save the dataframe to the specified path. Should end with '.csv'. Defaults to None.

        Raises:
            Exception: If the specified path to save the results does not work.
            Exception: If fit() was not conducted with compute_wrt_observation= True and loss_based= True.

        Returns:
            Union[pd.Series, pd.DataFrame]: Either a pd.Series with Sensitivity FI scores per feature (if global) 
            or a pd.DataFrame of Sensitivity FI scores with feautures as columns and test observations as rows.
        """
        try:
            if local:
                if save_to_path is not None:
                    try:
                        if not os.path.exists(os.path.dirname(save_to_path)):
                            os.makedirs(os.path.dirname(save_to_path))
                        self.FI_local.to_csv(save_to_path)
                    except:
                        raise ValueError(
                            "The specified path does not work. The path should end with '.csv'.")
                return self.FI_local
            else:
                if save_to_path is not None:
                    try:
                        if not os.path.exists(os.path.dirname(save_to_path)):
                            os.makedirs(os.path.dirname(save_to_path))
                        self.FI_global.to_csv(save_to_path)
                    except:
                        raise ValueError(
                            "The specified path does not work. The path should end with '.csv'.")
                return self.FI_global

        except:
            raise Exception(
                "FI values are not available. Refit with compute_wrt_observation= True and loss_based= True.")

    def get_FE(self,
               local: bool = False,
               save_to_path: Optional[str] = None) -> Union[pd.Series, pd.DataFrame]:
        """
        Return estimates of Sensitivity FE values from the fit()-function.
        Sensitivity FE values esimate the sensitivity of the predictions on features. 
        In detail, they indicate the effect of slightly enhancing the feature on the prediction.
        Sensitivity FE values can also be negative.

        Args:
            local (bool, optional): Whether to average the results across test-samples. Defaults to False.
            save_to_path (Optional[str], optional): If provided, save the dataframe to the specified path. Should end with '.csv'. Defaults to None.

        Raises:
            Exception: If the specified path to save the results does not work.
            Exception: If fit() was not conducted with compute_wrt_observation= True and pred_based= True.

        Returns:
            Union[pd.Series, pd.DataFrame]: Either a pd.Series with Sensitivity FE scores per feature (if global) 
            or a pd.DataFrame of Sensitivity FE scores with features as columns and test observations as rows.
        """
        try:
            if local:
                if save_to_path is not None:
                    try:
                        if not os.path.exists(os.path.dirname(save_to_path)):
                            os.makedirs(os.path.dirname(save_to_path))
                        self.FE_local.to_csv(save_to_path)
                    except:
                        raise ValueError(
                            "The specified path does not work. The path should end with '.csv'.")
                return self.FE_local
            else:
                if save_to_path is not None:
                    try:
                        if not os.path.exists(os.path.dirname(save_to_path)):
                            os.makedirs(os.path.dirname(save_to_path))
                        self.FE_global.to_csv(save_to_path)
                    except:
                        raise ValueError(
                            "The specified path does not work. The path should end with '.csv'.")
                return self.FE_global
        except:
            raise Exception(
                "FE values are not available. Refit with compute_wrt_observation= True and pred_based= False.")

    def get_OI(self,
               local: bool = False,
               save_to_path: Optional[str] = None) -> Union[pd.Series, pd.DataFrame]:
        """
        Return estimates of Sensitivity OI values from the fit()-function.
        Sensitivity OI values esimate the sensitivity of the predictive performance on training observations. 
        Sensitivity OI values are always positive.

        Args:
            local (bool, optional): Whether to average the results across test-samples. Defaults to False.
            save_to_path (Optional[str], optional): If provided, save the dataframe to the specified path. Should end with '.csv'. Defaults to None.

        Raises:            
            Exception: If the specified path to save the results does not work.
            Exception: If fit() was not conducted with compute_wrt_observation= True and pred_based= True.

        Returns:
            Union[pd.Series, pd.DataFrame]: Either a pd.Series with Sensitivity OI scores per train observation (if global) 
            or a pd.DataFrame of LOCO OI scores with train observations as columns and test observations as rows.
       """
        try:
            if local:
                if save_to_path is not None:
                    try:
                        if not os.path.exists(os.path.dirname(save_to_path)):
                            os.makedirs(os.path.dirname(save_to_path))
                        self.OI_local.to_csv(save_to_path)
                    except:
                        raise ValueError(
                            "The specified path does not work. The path should end with '.csv'.")
                return self.OI_local
            else:
                if save_to_path is not None:
                    try:
                        if not os.path.exists(os.path.dirname(save_to_path)):
                            os.makedirs(os.path.dirname(save_to_path))
                        self.OI_global.to_csv(save_to_path)
                    except:
                        raise ValueError(
                            "The specified path does not work. The path should end with '.csv'.")
                return self.OI_global
        except:
            raise Exception(
                "OI values are not available. Refit with compute_wrt_observation= True and loss_based= True.")

    def get_OE(self,
               local: bool = False,
               save_to_path: Optional[str] = None) -> Union[pd.Series, pd.DataFrame]:
        """
        Return estimates of Sensitivity OE values from the fit()-function.
        Sensitivity OE values estimate the sensitivity of the predictions on training observations. 
        Sensitivity OE values are always positive.

        Args:
            local (bool, optional): Whether to average the results across test-samples. Defaults to False.
            save_to_path (Optional[str], optional): If provided, save the dataframe to the specified path. Should end with '.csv'. Defaults to None.

        Raises:
            Exception: If the specified path to save the results does not work.
            Exception: If fit() was not conducted with compute_wrt_observation= True and pred_based= True.

        Returns:
            Union[pd.Series, pd.DataFrame]: Either a pd.Series with Sensitivity OE scores per train observation (if global) 
            or a pd.DataFrame of Sensitivity OE scores with train observations as columns and test observations as rows.
        """
        try:
            if local:
                if save_to_path is not None:
                    try:
                        if not os.path.exists(os.path.dirname(save_to_path)):
                            os.makedirs(os.path.dirname(save_to_path))
                        self.OE_local.to_csv(save_to_path)
                    except:
                        raise ValueError(
                            "The specified path does not work. The path should end with '.csv'.")
                return self.OE_local
            else:
                if save_to_path is not None:
                    try:
                        if not os.path.exists(os.path.dirname(save_to_path)):
                            os.makedirs(os.path.dirname(save_to_path))
                        self.OE_global.to_csv(save_to_path)
                    except:
                        raise ValueError(
                            "The specified path does not work. The path should end with '.csv'.")
                return self.OE_global
        except:
            raise Exception(
                "OE values are not available. Refit with compute_wrt_observation= True and pred_based= False.")

    def heatmap(self,
                plot_wrt_observation: bool = False,
                plot_pred_based: bool = False
                ):
        """
        Plots a heatmap of (up to 8) local Sensitiviy values and the global Sensitivity values.

        Args:
            plot_wrt_observation (bool, optional): Whether to plot the effects/importance of observations instead of features. Defaults to False.
            plot_pred_based (bool, optional): Whether to estimate feature/observation effects instead of importances. Defaults to False.

        Raises:
            Exception: If the queried configuration has not been fit.

        CAUTION: May lead to numerical errors for small datasets (hennce not tested in pytests).
        """
        ylabel = "Test Observation"

        try:
            if plot_wrt_observation:
                xlabel = "Training Observation"
                if not plot_pred_based:
                    colorbar_label = "Loss Sensitivity (OI)"
                    local_df = self.OI_local
                    global_df = self.OI_global
                else:
                    colorbar_label = "Prediction Sensitivity (OE)"
                    local_df = self.OE_local
                    global_df = self.OE_global
                plot_title = "Local and global " + colorbar_label + " per training observation"

            else:
                xlabel = "Feature"
                if not plot_pred_based:
                    colorbar_label = "Loss Sensitivity (FI)"
                    local_df = self.FI_local
                    global_df = self.FI_global
                else:
                    raise Exception(
                        "Heatmaps can not be plotted for FE, since they also contain negative values.")
                    # colorbar_label = "Prediction Sensitivity (FE)"
                    # local_df = self.FE_local
                    # global_df = self.FE_global
                plot_title = "Local and global " + colorbar_label + " per feature"

            plot_data = local_df
            min_LOCO = plot_data.min().min()
            max_LOCO = plot_data.max().max()
            vmax = np.max((np.abs(min_LOCO), max_LOCO))

            vmin = 0

            if plot_data.shape[0] > 8 or plot_data.shape[1] > 32:
                print(
                    "If the dataframe is to big, prune it to ensure that it can be plotted properly.")
                plot_data = plot_data.iloc[:8, :32]

            plot_data = pd.concat([plot_data, global_df], ignore_index=True)
            last_index = len(plot_data) - 1
            plot_data = plot_data.rename(index={last_index: 'Global'})
            plot_data.columns = [col[8:] for col in plot_data.columns]

            ax = sns.heatmap(plot_data,
                             vmin=vmin,
                             vmax=vmax,
                             center=0,
                             cmap=sns.diverging_palette(10, 220, as_cmap=True, sep=50))

            ax.collections[0].colorbar.set_label(colorbar_label)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(plot_title)

            plt.xticks(rotation=90)
            plt.gcf().set_size_inches(8, 6)  # Adjust the width and height as needed
            # Adjust the bottom margin so feature-labels are not cut off
            plt.subplots_adjust(bottom=0.5)

            plt.show()

        except:
            raise Exception(
                "The queried configuration has not been fit. Please refit correspondingly.")

    def boxplot(self,
                plot_wrt_observation: bool = False,
                plot_pred_based: bool = False
                ):
        """
        Plots a boxplot of local Sensitivity values.
        The boxplot are sorted by their median values.

        Args:
            plot_wrt_observation (bool, optional): Whether to plot the effects/importance of observations instead of features. Defaults to False.
            plot_pred_based (bool, optional): Whether to estimate feature/observation effects instead of importances. Defaults to False.

        Raises:
            Exception: If the queried configuration has not been fit.
        """
        try:
            if plot_wrt_observation:
                xlabel = "Training Observation"
                if not plot_pred_based:
                    ylabel = "Loss abs. Gradient"
                    local_df = self.OI_local
                else:
                    ylabel = "Prediction abs. Gradient"
                    local_df = self.OE_local
                plot_title = "Local " + ylabel + " per observation"

            else:
                xlabel = "Feature"
                if not plot_pred_based:
                    ylabel = "Loss abs. Gradient"
                    local_df = self.FI_local
                else:
                    ylabel = "Prediction Gradient"
                    local_df = self.FE_local
                plot_title = "Local " + ylabel + " per feature"

            plot_cols = [col for col in local_df.columns if col[:4] == "Sens"]

            plot_data = local_df[plot_cols]

            plot_data.columns = [col[8:] for col in plot_data.columns]

            if plot_data.shape[1] > 32:
                print("Pruned dataframe to ensure that it can be plotted properly.")
                plot_data = plot_data.iloc[:, :32]

            plot_data_melt = pd.melt(plot_data)
            medians = plot_data_melt.groupby('variable')['value'].median()
            plot_data_melt = plot_data_melt.merge(
                medians, left_on='variable', right_index=True, suffixes=('', '_median'))
            plot_data_melt = plot_data_melt.sort_values(by="value_median")

            # sort means in ascending order and assign a color palette
            if not plot_wrt_observation and plot_pred_based:
                palette = sns.diverging_palette(
                    10, 220, n=len(medians), sep=50)
                colors = pd.Series(
                    palette, index=medians.sort_values(ascending=True).index)
            else:
                palette = sns.color_palette(
                    "ch:s=.25,rot=-.25", n_colors=len(medians))
                colors = pd.Series(
                    palette, index=medians.sort_values(ascending=False).index)

            ax = sns.boxplot(x='variable',
                             y='value',
                             data=plot_data_melt,
                             palette=dict(colors),
                             showfliers=False)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(plot_title)

            plt.xticks(rotation=90)
            plt.gcf().set_size_inches(8, 6)  # Adjust the width and height as needed
            # Adjust the bottom margin so feature-labels are not cut off
            plt.subplots_adjust(bottom=0.5)

            plt.show()

        except:
            raise Exception(
                "The queried configuration has not been fit. Please refit correspondingly.")

    def plot_histogram(self,
                       plot_wrt_observation: bool = False,
                       plot_pred_based: bool = False,
                       bins: int = 20,
                       log_scale: List[bool] = [False, True]
                       ):
        """
        Plots a histogram of global (across all test samples) Sensitivity values of the model predictions or loss w.r.t features or train samples.

        Args:
            plot_wrt_observation (bool, optional): Whether to plot the Sensitivity w.r.t observations instead of features. Defaults to False.
            plot_pred_based (bool, optional): Whether to plot the Sensitivity of predictions instead of the model performance. Defaults to False.
            bins (int, optional): The amount of bins for the histogram. Defaults to 20.
            log_scale (List[bool], optional): For each axis (x and y), whether its scale is set to log. Defaults to [False, True].

        Raises:
            Exception: If the queried configuration has not been fit.
        """
        try:
            if plot_wrt_observation:
                name_indep = "Training Observations"
                if not plot_pred_based:
                    name_dep = "Loss"
                    global_data = self.OI_global.values
                else:
                    name_dep = "Prediction"
                    global_data = self.OE_global.values
            else:
                name_indep = "Features"
                if not plot_pred_based:
                    name_dep = "Loss"
                    global_data = self.FI_global.values
                else:
                    name_dep = "Prediction"
                    global_data = self.FE_global.values

            x_label = "Sensitivity of " + name_dep + " w.r.t " + name_indep

            hist_df = pd.DataFrame(global_data, columns=[x_label])
            hist_df = hist_df[np.isfinite(hist_df[x_label])]

            sns.histplot(data=hist_df,
                         x=x_label,
                         bins=bins,
                         kde=False,
                         log_scale=log_scale)
            plt.title("Histogram")
            plt.show()

        except:
            raise Exception(
                "The queried configuration has not been fit. Please refit correspondingly.")
