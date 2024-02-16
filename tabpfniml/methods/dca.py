import os
from datetime import datetime
from dcurves import dca
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from tabpfniml.methods.interpret import TabPFN_Interpret
from tabpfniml.datasets.datasets import dataset_iml
from typing import Optional, Iterable


class DCA(TabPFN_Interpret):
    """
    Provides an interface for Decision Curve Analysis using the dcurves package by Daniel D. Sjoberg.
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
            data (ConsultingData, optional): The dataset that TabPFN's behavior shall be explained for. Defaults to SpanishData.
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
                         store_gradients=False,
                         standardize_features=False)

    def fit(self,
            marker: str = None,
            random_forest: bool = False,
            gradient_boosting: bool = False,
            lightgbm: bool = False,
            ascertain_association: bool = False,
            seed: int = 42):
        """
        Fits the indicated models for comparison (no HPO), also fits TabPFN.
        If 'ascertain_association' is set to True, a logistic regression is fitted with the target as the dependent variable
        and the marker as independent variable in order to ascertain an association between the marker and the target.
        Coefficient significance is reported in the console output and the complete summary of the logistic regression can 
        be accessed via <DCAobject>.logreg_results.summary().

        Args:
            marker (str): Specifies a feature that is considered predictive of the probability/risk that the target equals 1. Defaults to None.
            random_forest (bool): If True, a Random Forest Classifier is fitted as a comparison to the other predictors. Defaults to False.
            gradient_boosting (bool): If True, a Gradient Boosting Classifier is fitted as a comparison to the other predictors. Defaults to False.
            lightgbm (bool): If True, a LightGBM Classifier is fitted as a comparison to the other predictors. Defaults to False.
            ascertain_association (bool): If True, a logistic regression is fitted with the target as dep. var. and the marker as indep. var. Defaults to False.
            seed (int, optional): Specifies the random seed for the classifiers. Defaults to 42.
        """

        self.marker = marker
        self.df_dca = pd.concat(
            [self.data.y_test_df.copy(), self.data.X_test_df.copy()], axis=1)

        if random_forest:
            rf_clf = RandomForestClassifier(random_state=seed)
            rf_clf.fit(self.X_train, self.y_train)
            rf_clf_preds = rf_clf.predict_proba(self.X_test)[:, 1]
            self.df_dca['Random Forest'] = rf_clf_preds

        if gradient_boosting:
            col_trans = make_column_transformer(
                (OneHotEncoder(handle_unknown="ignore"), self.data.categorical_features), remainder="passthrough")
            gb_clf = make_pipeline(
                col_trans, GradientBoostingClassifier(random_state=seed))
            gb_clf.fit(self.data.X_train_df, self.data.y_train_df)
            gb_clf_preds = gb_clf.predict_proba(self.data.X_test_df)[:, 1]
            self.df_dca['Gradient Boosting'] = gb_clf_preds

        if lightgbm:
            col_trans = make_column_transformer(
                (OneHotEncoder(handle_unknown="ignore"), self.data.categorical_features), remainder="passthrough")
            lgb_clf = make_pipeline(col_trans, LGBMClassifier(
                use_missing=True, objective="binary", random_state=seed))
            lgb_clf.fit(self.data.X_train_df, self.data.y_train_df)
            lgb_clf_preds = lgb_clf.predict_proba(self.data.X_test_df)[:, 1]
            self.df_dca['LightGBM'] = lgb_clf_preds

        self.classifier.fit(self.X_train, self.y_train)
        tabpfn_preds = self.classifier.predict_proba(self.X_test)[:, 1]
        self.df_dca['TabPFN'] = tabpfn_preds

        if ascertain_association:
            if self.marker == None:
                raise Exception(
                    'Association of outcome and marker can not be asserted, since no marker is specified.')
            form = self.data.y_df.name + ' ~ ' + marker
            logreg = sm.GLM.from_formula(
                form, data=self.data.df, family=sm.families.Binomial())
            self.logreg_results = logreg.fit()
            # 0 is for the intercept, 1 is for the covariate
            self.p_value = self.logreg_results.pvalues[1]
            self.is_significant = self.p_value < 0.05
            if self.is_significant:
                print("-" * 50)
                print(
                    f"The fitted logistic regression with dep. var. {self.data.y_df.name} and indep. var. {self.marker} has a significant coefficient.")
                print(
                    f"Thus, an association of {self.data.y_df.name} and {self.marker} is ascertained.")

    def add_predictor(self,
                      predictor_name: str,
                      predictor: object):
        """
        Allows the addition of additional predictive models. The only requirement is that the models yield predictions
        via a .predict() method, as these are needed to compute the net benefit values and the corresponding decision curve.

        Make sure to import the package of the predictor class! Example: from sklearn.neighbor import KNeighborsClassifier.

        Args:
            predictor_name (str): Specifies the name of the additional predictor as a string. Example: "kNN".
            predictor (object): Specifies an instance of a classifier. Example: KNeighborsClassifier() or RandomForestClassifier().
        """
        if not hasattr(self, 'df_dca'):
            raise AttributeError(
                'Make sure to first call .fit() on the DCA object before calling .add_predictor().')

        try:
            predictor.fit(self.data.X_train_df, self.data.y_train_df)
            predictions = predictor.predict_proba(self.data.X_test_df)[:, 1]
            self.df_dca[predictor_name] = predictions
        except:
            raise Exception(
                'Could not fit additional predictor. Make sure the predictor has a valid .predict() method and the respective imports are specified.')

    def plot(self,
             predictors: list,
             y_limits: list = [-0.05, 1.05],
             save_to_dir: Optional[str] = None):
        """
        Plots the decision curve for the desired predictors. In a DCA plot, the x-axis represents the threshold probability, 
        which is the probability at which a patient or clinician would opt for treatment. The y-axis represents the net benefit,
        a measurement calculated by summing the benefits (true positives) and subtracting the harms (false positives), appropriately
        weighted.

        The net benefit is calculated for each model across a range of threshold probabilities and plotted on the DCA plot. 
        The model with the highest net benefit at a given threshold probability is the most suitable for use at that threshold.

        Args:
            predictors (list): Specifies a list of predictors that should be included in the decision curve plot. For the marker in the '.fit()' method, its names need to be specified as they are spelled in the '.fit()' method. The other models are spelled like this: TabPFN, Random Forest, Gradient Boosting. Example: ["TabPFN", "Random Forest"]
            y_limits (list, optional): Specifies the range of the y-axis. Defaults to [-0.05, 1.05].
            save_to_dir (str, optional): If provided, save the plots to the specified directory.
        """
        if not hasattr(self, 'df_dca'):
            raise AttributeError(
                'Make sure to first call .fit() on the DCA object before calling .plot_dca().')

        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                  'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

        models_to_prob = []

        # Loop through all the model names
        for predictor in predictors:
            # Check if the model name is a column in the dataframe
            if predictor not in self.df_dca.columns:
                # Check if all the values in the column are between 0 and 1
                # if not self.df_dca[predictor].between(0, 1).all():
                 #If not, add the model name to the list
                models_to_prob.append(predictor)

        try:
            dca_risks = dca(
                data=self.df_dca,
                outcome=self.data.y_df.name,
                modelnames=predictors,
                models_to_prob=models_to_prob
            )
        except:
            raise Exception(
                'Either you specified predictors that were not specified in the .fit() method or the dcurves package could not be found. Please make sure it is installed in the current environment.')

        self.dca_risks = dca_risks

        ## TAKEN FROM THE DCURVES PACKAGE AND ADAPTED ###########################

        def _plot_net_benefit(
            plot_df: pd.DataFrame,
            y_limits: Iterable = (-0.05, 1),
            color_names: Iterable = None
        ) -> None:
            """
            Plot net benefit values against threshold values.

            Parameters
            ----------
            plot_df : pd.DataFrame
                Data containing threshold values, model columns of net benefit scores
                to be plotted
            y_limits : list[float]
                2 floats, lower and upper bounds for y-axis
            color_names : list[str]
                Colors to render each model (if n models supplied, then need n+2 colors,
                since 'all' and 'none' models will be included by default

            Returns
            -------
            None
            """

            modelnames = plot_df["model"].value_counts().index
            color_names = sns.color_palette('husl', n_colors=len(modelnames))
            for modelname, color_name in zip(modelnames, color_names):
                single_model_df = plot_df[plot_df["model"] == modelname]
                sns.lineplot(x='threshold', y='net_benefit',
                             data=single_model_df, color=color_name, label=modelname)
            plt.ylim(y_limits)
            plt.xlim(0, 1)
            plt.legend()
            plt.grid(
                color='black',
                which="both",
                axis="both",
                linewidth="0.3")
            plt.xlabel("Threshold Values")
            plt.ylabel("Net Benefit")
            plt.title("Net Benefit Curves")

            if save_to_dir:
                if not os.path.exists(save_to_dir):
                    os.makedirs(save_to_dir)
                plt.savefig(save_to_dir + "/dca_" + datetime.now().strftime(
                    "%Y-%m-%d-%H-%M-%S") + ".pdf", format="pdf")

            plt.show()

        def plot_graphs(
            plot_df: pd.DataFrame,
            graph_type: str = "net_benefit",
            y_limits: Iterable = (-0.05, 1),
            color_names: Optional[Iterable] = None,
        ) -> None:
            """
            Plot either net benefit or interventions avoided per threshold.

            Parameters
            ----------
            plot_df : pd.DataFrame
                Data containing threshold values, model columns of net benefit/intervention
                scores to be plotted
            graph_type : str
                Type of plot (either 'net_benefit' or 'net_intervention_avoided')
            y_limits : Iterable[Lower Bound, Upper Bound]
                2 floats, lower and upper bounds for y-axis
            color_names : Iterable[str]
                Colors to render each model (if n models supplied, then need n+2 colors,
                since 'all' and 'none' models will be included by default

            Returns
            -------
            None
            """

            if graph_type not in ["net_benefit", "net_intervention_avoided"]:
                raise ValueError(
                    "graph_type must be one of 2 strings: net_benefit,"
                    " net_intervention_avoided"
                )

            if graph_type == "net_benefit":
                _plot_net_benefit(
                    plot_df=plot_df, y_limits=y_limits, color_names=color_names)

        plot_graphs(
            plot_df=dca_risks,
            graph_type="net_benefit",
            y_limits=y_limits,
            color_names=None
        )

        ##############################################################
