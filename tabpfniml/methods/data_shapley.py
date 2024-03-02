import pandas as pd
import numpy as np
import torch
import math
from scipy.special import comb
import statsmodels.api as sm
from tabpfniml.methods.interpret import TabPFN_Interpret
from tabpfniml.datasets.datasets import OpenMLData
from sklearn.metrics import accuracy_score
from typing import Optional, Union, List, Dict
import os
import pickle


class Data_Shapley(TabPFN_Interpret):
    """
    Implementation of Data Shapley, a global data valuation method, which involves utilizing a variant of Kernel SHAP to attribute the empirical loss (on a validation set) to individual training observations.

    It can also be used for context optimization. In this case, the influence of training observations (for a training set of size > 1024) on empirical loss of TabPFN on a validation set is explained and used to determine an optimal subset.
    A further test set can then be used to obtain an unbiased performance estimate of the optimized training set.

    Currently only implemented for OpenML datasets!
    Currently only implemented for optimize_context= True!
    """

    def __init__(self,
                 optimize_context: bool = True,
                 openml_id: int = 819,
                 n_train: int = 1024,
                 n_val: int = 512,
                 n_test: int = 512,
                 seed: int = 728,
                 N_ensemble_configurations: int = 16,
                 device: str = "cpu",
                 debug: bool = False):
        """
        Initialize a TabPFN-interpretability method by passing rudimentary objects and the general configuration that is constant across all flavors of the interpretability-method.
        For some variables it may be permitted to overwrite them in fit()-methods, although this is not intended in general.

        It is also specified whether Data Shapley is employed for context optimization. If so, the data is split into three sets, where the predictions on the validation set are
        used to explain the influence of observations from the train set.

        Args:
            optimize_context (bool, optional): Whether the context/training data shall be optimized to maximize TabPFNs performance on the dataset. If True, it requires an additional validation set. Defaults to True.
            openml_id (int, optional): The OpenML dataset that TabPFN's behavior shall be explained for. Defaults to 770. Currently only implemented for OpenML datasets.
            n_train (int, optional): The amount of train-samples whose influence on TabPFNs performance shall be explained. May also be larger than 1024, if context optimization is conducted. Defaults to 1024.
            n_val (int, optional):  The size of a validation set on which TabPFNs performance shall be explained. Defaults to 512.
            n_test (int, optional): Conditional hyperparameter only considered if optimize_context= True. It specifies the size of a test set that is used to obtain an unbiased performance estimate for an optimized training set on TabPFN. Defaults to 512.
            seed (int, optional): Random seed used to shuffle the dataset and sample observation coalitions. Defaults to 728.
            N_ensemble_configurations (int, optional): The amount of TabPFN forward passes with different augmentations ensembled. Defaults to 16.
            device (str, optional): The device to store tensors and the TabPFN model on. Defaults to "cpu".
            debug (bool, optional): Whether debug mode is activated. This leads to e.g. less train and test samples and can hence tremendously reduce computational cost. Overwrites various other parameters. Defaults to False.

        Raises:
            Exception: If the specified training set size is smaller than 1024 and optimize_context= True. In this case, context optimization is not necessary.
            Exception: If the specified OpenML has to few entries for the specified dataset sizes.
        """
        self.n_train = n_train
        self.optimize_context = optimize_context

        self.seed = seed
        np.random.seed(self.seed)

        if self.optimize_context:
            n_test_full = n_val + n_test
            data_init = OpenMLData(openml_id,
                                   avoid_pruning=True,
                                   seed=self.seed)

        else:
            n_test_full = n_test
            data_init = OpenMLData(openml_id,
                                   avoid_pruning=False,
                                   seed=self.seed)

        if data_init.num_samples < (self.n_train + n_test_full):
            raise ValueError(
                "The specified OpenML has to few entries for the specified dataset sizes.")

        super().__init__(data=data_init,
                         n_train=self.n_train,
                         n_test=n_test_full,
                         N_ensemble_configurations=N_ensemble_configurations,
                         device=device,
                         debug=debug,
                         standardize_features=False,
                         to_torch_tensor=False,
                         store_gradients=False)

        if self.optimize_context:
            self.X_val = self.X_test[:n_val, :].copy()
            self.y_val = self.y_test[:n_val].copy()
            self.X_test = self.X_test[n_val:, :].copy()
            self.y_test = self.y_test[n_val:].copy()

    def fit(self,
            M_factor: int = 1,
            tPFN_train_min: int = 128,
            tPFN_train_max: int = 1024,
            class_to_be_explained: int = 1
            ):
        """
        Fits a local surrogate model (weighted linear model) to predict the empirical loss for validation samples given observation coalitions.
        The resulting surrogate model coefficients are interpreted as local Shapley values (as the loss is explained, low values are expected to enhance the predictive performance).
        M* M_factor observation coalitions are used to estimate the effect of observations on the model performance, where each observation coalition has a size between tPFN_train_min and tPFN_train_max.

        Args:
            M_factor (int, optional): An integer value to specify how many observation coalitions shall be considered, where the amount of coalitions is n_train * M_factor. Must at least be 1 to avoid n<p. Defaults to 1.
            tPFN_train_min: Minimal training set size for TabPFN forward passes to get meaningful predicitons. Serves as the lower bound for the size of the observation coalitions. Defaults to 128.
            tPFN_train_max: Maximum training set size for TabPFN forward passes. Serves as the upper bound for the size of the observation coalitions. Defaults to 1024, as proposed by the authors.
            class_to_be_explained (int, optional): The class that predictions are explained for. Ignored it pred_based=False. Defaults to 1.

        Raises:
            Exception: If init() has been called with optimize_context=False as this is not implemented yet.
        """
        def get_kernel_weight(cs: int, p: int) -> np.float64:
            """
            Compute weights for coalitions through the kernel.
            Small and large coalitions get the highest weights.

            Args:
                cs (int): The amount of observations in the coalition.
                p (int): The total amount of observations.

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

        self.M_factor = M_factor
        self.tPFN_train_min = tPFN_train_min
        self.tPFN_train_max = tPFN_train_max
        self.class_to_be_explained = class_to_be_explained

        if self.optimize_context:
            if self.n_train < self.tPFN_train_max:
                raise ValueError(
                    "There is no need for context optimization, since the entire dataset fits in a single TabPFN  forward pass.")

        self.M = int(self.n_train * self.M_factor)
        assert self.X_train.shape[0] == self.n_train
        self.train_indices = list(range(self.n_train))

        design_matrix = pd.DataFrame()
        weights = pd.Series()
        loss_values = pd.DataFrame()
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        if self.optimize_context:
            coal_p = float(
                (self.tPFN_train_min + (self.tPFN_train_max-self.tPFN_train_min)*0.5)
                / self.tPFN_train_max
                / (self.n_train / self.tPFN_train_max)
            )

            def get_coal():
                coalition_mask = np.random.binomial(
                    n=1, p=coal_p, size=self.n_train)
                coalition_mask_bool = np.array(coalition_mask, dtype=bool)
                return coalition_mask, coalition_mask_bool

            for m in range(self.M):
                print(str(m) + "/" + str(self.M))
                print("---")
                # Step 1: Sample M observation coalitions
                coalition_mask, coalition_mask_bool = get_coal()

                # Case that size of sampled coalition exceeds training set bounds
                while sum(coalition_mask) < self.tPFN_train_min or sum(coalition_mask) > self.tPFN_train_max:
                    coalition_mask, coalition_mask_bool = get_coal()

                weight = get_kernel_weight(
                    coalition_mask.sum(), self.tPFN_train_max)

                # Step 2: Transfer coalitions into observations space and get predictions by applying TabPFN.
                X_train_coal = self.X_train[coalition_mask_bool, :].copy()
                y_train_coal = self.y_train[coalition_mask_bool].copy()

                self.classifier.fit(X_train_coal, y_train_coal)
                preds = self.classifier.predict_proba(self.X_val.copy())
                loss = self.criterion(torch.tensor(preds), torch.tensor(
                    self.y_val.copy(), dtype=torch.long))

                loss_values = pd.concat([loss_values, pd.DataFrame([loss.item()])],
                                        ignore_index=True,
                                        axis=0)

                design_matrix = pd.concat([design_matrix, pd.DataFrame([coalition_mask])],
                                          ignore_index=True,
                                          axis=0)

                weights = pd.concat([weights, pd.Series(weight)],
                                    ignore_index=True,
                                    axis=0)

            # Step 4: Fit a weighted linear model
            # Also, we track the performance of context optimization w.r.t m
            design_matrix = sm.add_constant(design_matrix, prepend=True)

            self.data_values = {}
            self.m_range = [2**x for x in range(4, 16)]
            if self.M not in self.m_range:
                self.m_range.append(self.M)

            for m in self.m_range:
                if m >= 16 and m <= self.M:
                    lm = sm.WLS(loss_values.iloc[0:m],
                                design_matrix.iloc[0:m, :],
                                weights=weights.iloc[0:m])
                    self.lm_res = lm.fit()

                    # Step 5: Extract Data values values
                    self.data_values[m] = pd.Series(
                        np.array(self.lm_res.params[1:]), index=self.train_indices)
                else:
                    pass

        else:
            raise ValueError("Not implemented yet.")

    def get_data_values(self,
                        save_to_path: Optional[str] = None) -> pd.Series:
        """
        Returns data values estimated in the fit()-function.
        Data values assign the contribution of each training observation to the empirical loss on the validation set.
        Therefore, low data values are expected to enhance the predictive performance.

        Args:
            save_to_path (Optional[str], optional): If provided, save the dataframe to the specified path. Should end with '.csv'. Defaults to None.

        Raises:
            Exception: If fit() has been applied with loss_based=True instead of loss_based=False or not at all.
            Exception: If the specified path does not work.

        Returns:
            pd.Series: Series of data values.
        """
        try:
            if save_to_path is not None:
                try:
                    if not os.path.exists(os.path.dirname(save_to_path)):
                        os.makedirs(os.path.dirname(save_to_path))
                    self.data_values[self.M].to_csv(save_to_path)
                except:
                    raise ValueError(
                        "The specified path does not work. The path should end with '.csv'.")
            return self.data_values[self.M]
        except:
            raise Exception(
                "To obtain data values, execute fit() first. At the moment, it is also required to call init() with optimize_context= True, since the alternative option is not implemented yet.")

    def get_optimized_context(self,
                              size=None) -> List:
        """
        Returns a list with the training set indices of the most relevant training observations according to Data Shapley.
        Together, they constitute an optimized context.

        Args:
            size (optional): The size of the optimized context. Defaults to None, setting it equal to self.tPFN_train_max.

        Raises:
            ValueError: If the queried context size is larger than the maximal amount of training observations for a TabPFN forward pass.
            Exception: If init(optimize_context= True) and fit() has not been called yet.

        Returns:
            List: The indices of the most relevant training observations.
        """
        try:
            if size != None:
                if size > self.tPFN_train_max:
                    raise ValueError(
                        "Optimized context cannot be larger than maximal amount of training observations for a TabPFN forward pass.")
            else:
                size = self.tPFN_train_max
            return self.data_values[self.M].nsmallest(size).index.tolist()

        except:
            raise Exception(
                "To obtain data values, execute init() with optimize_context= True and fit() first.")

    def get_optimized_performance_diff(self,
                                       save_to_path: Optional[str] = None) -> Dict:
        """
        Compare the performance of an optimized context to random contexts on the previously unseen test set.

        Args:
            save_to_path (Optional[str], optional): If provided, save the dict to the specified path. Should end with '.csv'. Defaults to None.

        Raises:
            Exception: If the specified path to save the results does not work.
            Exception: If init(optimize_context= True) and fit() has not been called yet.

        Returns:
            Dict: Dict containing information about the performance difference of optimized and random contexts on the test set.
        """

        try:
            random_losses = []
            random_accs = []

            results_dict = {"seed": [],
                            "M": [],
                            "RC Mean Loss": [],
                            "RC Mean Acc": [],
                            "RC Std Loss": [],
                            "RC Std Acc": [],
                            "OC Loss": [],
                            "OC Acc": []}

            # Obtain Loss and Accuracy for multiple random contexts
            amount_random_train_sets = math.floor(
                self.n_train / self.tPFN_train_max)
            for i in range(amount_random_train_sets):
                temp_X_train = self.X_train[i * self.tPFN_train_max:
                                            (i + 1) * self.tPFN_train_max, :].copy()
                temp_y_train = self.y_train[i * self.tPFN_train_max:
                                            (i + 1) * self.tPFN_train_max].copy()

                self.classifier.fit(temp_X_train, temp_y_train)
                temp_preds = self.classifier.predict_proba(self.X_test.copy())

                temp_loss = self.criterion(torch.tensor(temp_preds), torch.tensor(
                    self.y_test.copy(), dtype=torch.long))

                temp_preds_hard = self.classifier.classes_.take(
                    np.asarray(np.argmax(temp_preds, axis=-1), dtype=np.intp))
                temp_acc = accuracy_score(torch.tensor(
                    self.y_test.copy(), dtype=torch.long), temp_preds_hard)

                random_losses.append(temp_loss.item())
                random_accs.append(temp_acc)

            # Obtain Loss and Accuracy for optimized contexts
            for m in self.m_range:
                if m >= 16 and m <= self.M:
                    opt_indices = self.data_values[m].nsmallest(
                        self.tPFN_train_max).index.tolist()
                    X_train_opt = self.X_train[opt_indices, :].copy()
                    y_train_opt = self.y_train[opt_indices].copy()

                    self.classifier.fit(X_train_opt, y_train_opt)
                    preds_opt = self.classifier.predict_proba(
                        self.X_test.copy())

                    loss_opt = self.criterion(torch.tensor(preds_opt), torch.tensor(
                        self.y_test.copy(), dtype=torch.long))

                    preds_opt_hard = self.classifier.classes_.take(
                        np.asarray(np.argmax(preds_opt, axis=-1), dtype=np.intp))
                    acc_opt = accuracy_score(torch.tensor(
                        self.y_test.copy(), dtype=torch.long), preds_opt_hard)

                    results_dict["seed"].append(self.seed)
                    results_dict["M"].append(m)
                    results_dict["RC Mean Loss"].append(np.mean(random_losses))
                    results_dict["RC Mean Acc"].append(np.mean(random_accs))
                    results_dict["RC Std Loss"].append(np.std(random_losses))
                    results_dict["RC Std Acc"].append(np.std(random_accs))
                    results_dict["OC Loss"].append(loss_opt.item())
                    results_dict["OC Acc"].append(acc_opt)
                    # "RC Detailed_Loss": random_losses,
                    # "RC Detailed_Acc": random_accs,

            results_df = pd.DataFrame.from_dict(results_dict)

            if save_to_path is not None:
                try:
                    if not os.path.exists(os.path.dirname(save_to_path)):
                        os.makedirs(os.path.dirname(save_to_path))

                    results_df.to_csv(save_to_path, index=True)
                except:
                    raise ValueError(
                        "The specified path does not work. The path should end with '.csv'.")

            return results_df

        except:
            raise Exception(
                "To get the performance difference, execute init() with optimize_context= True and fit() first.")
