import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabpfniml.methods.interpret import TabPFN_Interpret
from tabpfniml.datasets.datasets import dataset_iml
from mapie.classification import MapieClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Optional
import os
from datetime import datetime
import time


class Conformal_Prediction(TabPFN_Interpret):
    """
    Provides conformal prediction intervals and plotting options for multiclass classification using the mapie package.

    Suppose we want to classify whether patients are at risk of developing one of 10 (mutually exclusive) medical conditions.
    We have trained a classifier on this multiclass classification task and the model provides us with predicted probabilities
    (softmax output) for our test set. Conformal Prediction is a straightforward way to generate prediction sets
    (a set of possible labels) that satisfy 1 - alpha coverage for every observation in the test using a small amount of (additional)
    calibration data that, in this implementation, is taken as a subset of the train set.
    For a good theoretical introduction see Angelopoulos & Bates (2021): https://doi.org/10.48550/arXiv.2107.07511
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
            data (dataset_iml, optional): The dataset that TabPFN's behavior shall be explained for. Defaults to SpanishData.
            n_test (int, optional): The amount of train-samples to fit TabPFN on. Should not be larger than 1024. Defaults to 512.
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
                         standardize_features= False,
                         store_gradients= False,
                         to_torch_tensor=False)

    def fit(self,
            alpha: float = 0.05,
            cv: str = "prefit",
            method: str = "score",
            train_calib_size: float = 0.7,
            seed=42) -> None:
        """
        Fits the TabPFN classifier and the MapieClassifier from the mapie python package to the data.
        The calibration set is constructed as a subset of the train set. 
        Classifiers are only trained on the remaining part of the train set. 

        Args:
            alpha (float, optional): The significance level. For a 95% prediction set, choose alpha=0.05. Defaults to 0.05.
            cv (str, optional): The cv method of the MapieClassifier. Defaults to "prefit".
            method (str, optional): The score method of the MapieClassifier. Defaults to "score".
            train_calib_size (float, optional): Specifies how much of the training set should be used as a calibration set to fit
            the MapieClassifier. Float between 0 and 1, represents tradeoff between train and calibration set, e.g., 0.7 means
            that 30% of the train set are used as a calibration set. Defaults to 0.7.
            seed (int, optional): The random_state of the train-calibration set split. Defaults to 42.
        """
        self.alpha = alpha
        self.cv = cv
        self.method = method

        self.X_train, self.X_calib, self.y_train, self.y_calib = train_test_split(
            self.X_train, self.y_train, train_size=train_calib_size, random_state=seed)

        self.classifier.fit(self.X_train, self.y_train)
        try:
            self.cp = MapieClassifier(
                estimator=self.classifier, cv=self.cv, method=self.method)
            self.cp.fit(self.X_calib, self.y_calib)
        except:
            raise Exception(
                "Could not instantiate MapieClassifier. Make sure you have installed the mapie python package in your environment.")

    def get_conformal_prediction(self,
                                 save_to_path: Optional[str] = None) -> pd.DataFrame:
        """
        Returns prediction sets (a set of possible labels) that satisfy 1 - alpha coverage for every observation
        in the test using a small amount of (additional) calibration data that, in this implementation, 
        is taken as a subset of the train set.

        The prediction sets and predictions are accessible via the fields/arguments y_set and y_pred.

        Args:
            save_to_path (Optional[str], optional): If provided, save the dataframe to the specified path. Should end with '.csv'. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the prediction sets for every test observation.
        """
        try:
            if not hasattr(self, 'cp'):
                raise AttributeError(
                    'Fitted classifier not found. Make sure to first call .fit() on the object before calling .conformal_prediction().')

            y_pred, y_set = self.cp.predict(self.X_test, alpha=self.alpha)
            self.y_pred = y_pred
            self.y_set = y_set

            y_set = np.squeeze(y_set)

            y_preds, y_pred_sets = [], []
            for i in range(self.y_test.shape[0]):
                y_preds.append(y_pred[i])
                y_pred_sets.append(self.classifier.classes_[y_set[i]])

            y_pred_sets = [l.tolist() for l in y_pred_sets]
            y_pred_sets_strings = [','.join(map(str, arr))
                                   for arr in y_pred_sets]
            pred_and_pred_set = pd.DataFrame(
                {'prediction': y_preds, f'prediction set: {int((1-self.alpha) * 100)}%': y_pred_sets_strings})

            if save_to_path is not None:
                try:
                    if not os.path.exists(os.path.dirname(save_to_path)):
                        os.makedirs(os.path.dirname(save_to_path))
                    pred_and_pred_set.to_csv(save_to_path)
                except:
                    raise ValueError(
                        "The specified path does not work. The path should end with '.csv'.")
            return pred_and_pred_set

        except:
            raise Exception(
                "To obtain conformal predictions, call the .fit() method on the conformal prediction object first.")

    def plot(self,
            alphas: list,
            x_axis: int = 0,
            y_axis: int = 1,
            point_size: int = 30,
            opacity_and_size: bool = True, 
            save_to_dir: Optional[str] = None):
        """
        Plots grid of plots that show the scatterplots of the predicted class of the test observations.
        On the scatterplots' x-axis and y-axis, features of the dataset are displayed that can be changed
        using the `x_axis` and `y_axis` arguments.
        For each alpha level (significance level) specified in the alphas argument, one plot is added 
        that shows how many different labels each test observation's prediction set contains.
        If observations' prediction sets contain the same number of labels, the corresponding points are plotted
        with the same opacity.

        Args:
            alphas (list): List of up to 3 alpha values, each value between 0 and 1. Example: [0.2, 0.1, 0.05]
            x_axis (int, optional): The column index of the feature that should be displayed on the x-axis. Defaults to 0.
            y_axis (int, optional): The column index of the feature that should be displayed on the y-axis. Defaults to 1.
            point_size (int, optional): The size of the points in the scatterplots. Defaults to 30.
            opacity_and_size (bool, optional): If True, the prediction set size of an observation is shown as difference in opacity
            and size, otherwise only as difference in opacity.
            save_to_dir (str, optional): If provided, save the plots to the specified directory.

        """

        if not hasattr(self, 'cp'):
            raise AttributeError(
                'Make sure to first call .fit() on the object before attempting to plot.')

        y_pred, y_set = self.cp.predict(self.X_test, alpha=alphas) 
        X = self.X_test
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()


        # colors = {0: 'red', 1: 'green', 2:  'blue', 3: 'cyan', 4: 'magenta', 5: 'yellow', 6: 'black', 7: 'purple', 8: 'pink', 9: 'brown', 10: 'grey'}
        # all_colors = {1: '#440154', 2: '#482878', 3: '#3e4a89', 4: '#31688e', 5: '#26828e', 6: '#1f9e89', 7: '#35b779', 8: '#6dcd59', 9: '#b8de29', 10: '#fde725'}
        # colors = list(all_colors.values())
        # colors = plt.cm.get_cmap('viridis', self.data.y_classes)
        # y_pred_col = list(map(colors, y_pred))

        if len(alphas) == 3:
            fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(
                2, 2, figsize=(10, 10))
            axs = {0: ax1, 1: ax2, 2: ax3, 3: ax4}
        elif len(alphas) == 2:
            fig, [ax1, ax2, ax3] = plt.subplots(3, figsize=(10, 10))
            axs = {0: ax1, 1: ax2, 2: ax3}
        elif len(alphas) == 1:
            fig, [ax1, ax2] = plt.subplots(2, figsize=(10, 10))
            axs = {0: ax1, 1: ax2}
        else:
            raise ValueError("The number of alphas is not between 1 and 3.")

        axs[0].scatter(
            X[:, x_axis],
            X[:, y_axis],
            marker='.',
            s=point_size,
            c=y_pred.astype(int),
            cmap='viridis',
            alpha=0.4
        )
        axs[0].set_title("Predicted labels")

        for i, alpha in enumerate(alphas):
            y_set_sums = y_set[:, :, i].sum(axis=1)
            y_set_max = y_set_sums.max()
            if opacity_and_size:
                num_labels = axs[i+1].scatter(
                    X[:, x_axis],
                    X[:, y_axis],
                    c=y_set_sums,
                    marker='.',
                    s=y_set_sums * point_size,
                    alpha=1,
                    cmap='viridis',
                    vmin=0,
                    vmax=y_set_max
                )
            else:
                num_labels = axs[i+1].scatter(
                    X[:, x_axis],
                    X[:, y_axis],
                    c=y_set_sums,
                    marker='.',
                    s=point_size,
                    alpha=1,
                    cmap='viridis',
                    vmin=0,
                    vmax=y_set_max
                )
            colorbar = fig.colorbar(num_labels, ax=axs[i+1])
            colorbar.locator = ticker.MaxNLocator(integer=True)
            colorbar.update_ticks()
            axs[i+1].set_title(f"Number of labels for alpha={alpha}")
        
        if save_to_dir:
            if not os.path.exists(save_to_dir):
                os.makedirs(save_to_dir)
            
            plt.savefig(save_to_dir + "/conformal_pred_" + datetime.now().strftime(
                "%Y-%m-%d-%H-%M-%S") + ".pdf", format="pdf")
            time.sleep(1)