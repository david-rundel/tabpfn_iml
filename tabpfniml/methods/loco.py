
import pandas as pd
import numpy as np
import torch
from tabpfniml.methods.interpret import TabPFN_Interpret
from tabpfniml.datasets.datasets import dataset_iml
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Union, Optional
import os

class LOCO(TabPFN_Interpret):
    """
    Implementation of LOCO (Leave-One-Covariate-Out), an interpretability method to understand the importance of individual features on model performance. 
    It involves systematically removing one feature at a time and measuring the resulting change in performance.
    Since fitting TabPFN on new data is almost free in terms of computational demands, we also estimate the importance of individual observations in a 
    similar fashion by removing them and refitting the model.
    This basic idea is also used to obtain feature effect measures instead of feature importance measure, by analyzing the change in model predictions.
    """
    def __init__(self,
                 data: dataset_iml,
                 n_train: int= 1024,
                 n_test: int= 512,
                 N_ensemble_configurations: int= 16,
                 device: str= "cpu",
                 debug: bool= False):
        """
        Initialize a TabPFN-interpretability method by passing rudimentary objects and the general configuration that is constant across all flavors of the interpretability-method.
        For some variables it may be permitted to overwrite them in fit()-methods, although this is not intended in general.
        Compared to the parent-classes __init__()-method, it is ensured that the data is loaded as torch.tensor.
        
        Args:
            data (dataset_iml, optional): The dataset that TabPFN's behavior shall be explained for. Defaults to dataset_iml.
            n_train (int, optional): The amount of train-samples to fit TabPFN on. Should not be larger than 1024. Defaults to 512.
            n_test (int, optional): The amount of test-samples to get predictions for. Defaults to 512.
            N_ensemble_configurations (int, optional): The amount of TabPFN forward passes with different augmentations ensembled. Defaults to 16.
            device (str, optional): The device to store tensors and the TabPFN model on. Defaults to "cpu".
            debug (bool, optional): Whether debug mode is activated. This leads to e.g. less train and test samples and can hence tremendously reduce computational cost. Overwrites various other parameters. Defaults to False.
        """
        super().__init__(data= data,
                         n_train= n_train,
                         n_test= n_test,
                         N_ensemble_configurations= N_ensemble_configurations,
                         device= device,
                         debug= debug,
                         standardize_features= False,
                         store_gradients= False,
                         to_torch_tensor= True)

    def fit(self,
            compute_wrt_feature: bool= True,
            compute_wrt_observation: bool= False,
            n_train_relevance: int= 32,
            loss_based: bool= True,
            pred_based: bool= False,
            class_to_be_explained: int= 1
            ):
        """
        Estimate LOCO by removing features or samples and measuring the resulting change in model performance or predictions.

        Args:
            compute_wrt_feature (bool, optional): Whether to estimate the effects/importance of features. Defaults to True.
            compute_wrt_observation (bool, optional): Whether to estimate the effects/importance of training observations. Defaults to False.
            n_train_relevance (int, optional): The amount of training observations to estimate the effects/importance for. Ignored if compute_wrt_observation= False. Defaults to 32.
            loss_based (bool, optional): Whether to estimate feature/observation importances. Defaults to True.
            pred_based (bool, optional): Whether to estimate feature/observation effects. Defaults to False.
            class_to_be_explained (int, optional): The class that feature/observation effects are explained for. Ignored if pred_based= False. Defaults to 1.

        Raises:
            ValueError: If the requested amount of observations to estimate effects for (n_train_relevance) is greater than n_train.
        """
        self.compute_wrt_feature= compute_wrt_feature
        self.compute_wrt_observation= compute_wrt_observation
        self.loss_based= loss_based
        self.pred_based= pred_based
        self.class_to_be_explained= class_to_be_explained

        if self.debug:
            n_train_relevance= 4
        if n_train_relevance <= self.n_train:
            self.n_train_relevance= n_train_relevance
        else:
            raise ValueError("n_train_relevancy has to be smaller than or equal n_train.")

        self.baseline_name= "Baseline"

        self.classifier.fit(self.X_train, self.y_train)
        preds= torch.tensor(self.classifier.predict_proba(self.X_test))

        if self.loss_based:
            self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
            loss_X_train= self.criterion(preds, self.y_test).detach().numpy()

        #Step 1: Set up dataframes with baseline predictions/ predictive performances (all features and all values)
        if self.compute_wrt_feature:
            if self.loss_based:
                #LOCO as Feature Importance (FI)
                self.FI_local = pd.DataFrame()
                self.FI_local["Baseline"] = loss_X_train

            if self.pred_based:
                #LOCO as Feature Effects (FE)
                self.FE_local = pd.DataFrame()
                self.FE_local["Baseline"]= preds[:,self.class_to_be_explained].detach().numpy()

        if self.compute_wrt_observation:
            if self.loss_based:
                #LOCO as Observation Importance (OI)
                self.OI_local = pd.DataFrame()
                self.OI_local["Baseline"] = loss_X_train

            if self.pred_based:
                #LOCO as Observation Effect (OE)
                self.OE_local = pd.DataFrame()
                self.OE_local["Baseline"]= preds[:,self.class_to_be_explained].detach().numpy()

        # Step 2: Compute LOCO scores
        # Compute LOCO as FI/FE
        if self.compute_wrt_feature:
            for j in range(self.data.num_features):
                #Remove current feature
                subset_X_train= torch.cat((self.X_train[:,:j], self.X_train[:,j+1:]), dim=1)
                self.classifier.fit(subset_X_train, self.y_train)

                subset_X_test= torch.cat((self.X_test[:,:j], self.X_test[:,j+1:]), dim=1)
                temp_preds= torch.tensor(self.classifier.predict_proba(subset_X_test))

                if self.pred_based:
                    #LOCO as FE
                    self.FE_local["LOCO_FE_" + self.data.feature_names[j]]= (temp_preds[:,self.class_to_be_explained].detach().numpy() - self.FE_local[self.baseline_name]).abs()

                if self.loss_based:
                    #LOCO as FI
                    self.FI_local["LOCO_FI_" + self.data.feature_names[j]]= self.criterion(temp_preds, self.y_test).detach().numpy() - self.FI_local[self.baseline_name]


        #Compute LOCO as OI/OE
        if self.compute_wrt_observation:
            for i in range(self.n_train_relevance):
                #Remove current observation
                subset_X_train= torch.cat((self.X_train[:i,:], self.X_train[i+1:,:]), dim=0)
                subset_y_train= torch.cat((self.y_train[:i], self.y_train[i+1:]), dim=0)
                self.classifier.fit(subset_X_train, subset_y_train)
                temp_preds= torch.tensor(self.classifier.predict_proba(self.X_test))

                if self.pred_based:
                    #LOCO as OE
                    self.OE_local["LOCO_OE_" + str(i)]= (temp_preds[:,self.class_to_be_explained].detach().numpy() - self.OE_local[self.baseline_name]).abs()

                if self.loss_based:
                    #LOCO as OI
                    self.OI_local["LOCO_OI_" + str(i)]= self.criterion(temp_preds, self.y_test).detach().numpy() - self.OI_local[self.baseline_name]


        # Step 3: Compute global statistics
        if self.compute_wrt_feature:
            if self.pred_based:
                #LOCO as FE
                self.FE_global = self.FE_local[[col for col in self.FE_local.columns if col[:4]== "LOCO"]].mean()

            if self.loss_based:
                #LOCO as FI
                self.FI_global = self.FI_local[[col for col in self.FI_local.columns if col[:4]== "LOCO"]].mean()

        if self.compute_wrt_observation:
            if self.pred_based:
                #LOCO as OE
                self.OE_global = self.OE_local[[col for col in self.OE_local.columns if col[:4]== "LOCO"]].mean()

            if self.loss_based:
                #LOCO as OI
                self.OI_global = self.OI_local[[col for col in self.OI_local.columns if col[:4]== "LOCO"]].mean()

    def get_FI(self, 
               local: bool= False,
               save_to_path: Optional[str]= None) -> Union[pd.Series, pd.DataFrame]:
        """
        Return estimates of LOCO FI values from the fit()-function.
        LOCO FI values esimate the effect on predictive performance when removing individual features. 
        Negative values may indicate that a feature is detrimental to the performance.
        
        Args:
            local (bool, optional): Whether to average the results across test-samples. Defaults to False.
            save_to_path (Optional[str], optional): If provided, save the dataframe to the specified path. Should end with '.csv'. Defaults to None.
            
        Raises:
            Exception: If the specified path to save the results does not work.
            Exception: If fit() was not conducted with compute_wrt_observation= True and loss_based= True.

        Returns:
            Union[pd.Series, pd.DataFrame]: Either a pd.Series with LOCO FI scores per feature (if global) 
            or a pd.DataFrame of LOCO FI scores with feautures as columns and test observations as rows.
        """
        try:
        #if self.compute_wrt_feature and self.loss_based: (not suitable if fit() several times)
            if local:
                if save_to_path is not None:
                    try:
                        if not os.path.exists(os.path.dirname(save_to_path)):
                            os.makedirs(os.path.dirname(save_to_path))
                        self.FI_local.to_csv(save_to_path)
                    except:
                        raise ValueError("The specified path does not work. The path should end with '.csv'.") 
                return self.FI_local
            else:
                if save_to_path is not None:
                    try:
                        if not os.path.exists(os.path.dirname(save_to_path)):
                            os.makedirs(os.path.dirname(save_to_path))
                        self.FI_global.to_csv(save_to_path)
                    except:
                        raise ValueError("The specified path does not work. The path should end with '.csv'.") 
                return self.FI_global
        except:
            raise Exception("FI values are not available. Refit with compute_wrt_observation= True and loss_based= True.")
    
    def get_FE(self, 
               local: bool= False,
               save_to_path: Optional[str]= None) -> Union[pd.Series, pd.DataFrame]:
        """
        Return estimates of LOCO FE values from the fit()-function.
        LOCO FE values estimate the effect on predictions when removing individual features.
        Effect (instead of importance) values are always positive and their intensities indicate feature effects.
        
        Args:
            local (bool, optional): Whether to average the results across test-samples. Defaults to False.
            save_to_path (Optional[str], optional): If provided, save the dataframe to the specified path. Should end with '.csv'. Defaults to None.
            
        Raises:
            Exception: If the specified path to save the results does not work.
            Exception: If fit() was not conducted with compute_wrt_observation= True and pred_based= True.

        Returns:
            Union[pd.Series, pd.DataFrame]: Either a pd.Series with LOCO FE scores per feature (if global) 
            or a pd.DataFrame of LOCO FE scores with features as columns and test observations as rows.
         """
        try:
            if local:
                if save_to_path is not None:
                    try:
                        if not os.path.exists(os.path.dirname(save_to_path)):
                            os.makedirs(os.path.dirname(save_to_path))
                        self.FE_local.to_csv(save_to_path)
                    except:
                        raise ValueError("The specified path does not work. The path should end with '.csv'.") 
                return self.FE_local
            else:
                if save_to_path is not None:
                    try:
                        if not os.path.exists(os.path.dirname(save_to_path)):
                            os.makedirs(os.path.dirname(save_to_path))
                        self.FE_global.to_csv(save_to_path)
                    except:
                        raise ValueError("The specified path does not work. The path should end with '.csv'.") 
                return self.FE_global
            
        except:
            raise Exception("FE values are not available. Refit with compute_wrt_observation= True and pred_based= False.")

    def get_OI(self, 
               local: bool= False,
               save_to_path: Optional[str]= None) -> Union[pd.Series, pd.DataFrame]:
        """
        Return estimates of LOCO OI values from the fit()-function.
        LOCO OI values esimate the effect on predictive performance when removing individual observations. 
        Negative values may indicate that an observation is detrimental to the performance.
        
        Args:
            local (bool, optional): Whether to average the results across test-samples. Defaults to False.
            save_to_path (Optional[str], optional): If provided, save the dataframe to the specified path. Should end with '.csv'. Defaults to None.
            
        Raises:
            Exception: If the specified path to save the results does not work.
            Exception: If fit() was not conducted with compute_wrt_observation= True and loss_based= True.

        Returns:
            Union[pd.Series, pd.DataFrame]: Either a pd.Series with LOCO OI scores per train observation (if global) 
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
                        raise ValueError("The specified path does not work. The path should end with '.csv'.") 
                return self.OI_local
            else:
                if save_to_path is not None:
                    try:
                        if not os.path.exists(os.path.dirname(save_to_path)):
                            os.makedirs(os.path.dirname(save_to_path))
                        self.OI_global.to_csv(save_to_path)
                    except:
                        raise ValueError("The specified path does not work. The path should end with '.csv'.") 
                return self.OI_global
        except:
            raise Exception("OI values are not available. Refit with compute_wrt_observation= True and loss_based= True.")
     
    def get_OE(self, 
               local: bool= False,
               save_to_path: Optional[str]= None) -> Union[pd.Series, pd.DataFrame]:
        """
        Return estimates of LOCO OE values from the fit()-function.
        LOCO OE values estimate the effect on predictions when removing individual observations.
        Effect (instead of importance) values are always positive and their intensities indicate observation effects.
        
        Args:
            local (bool, optional): Whether to average the results across test-samples. Defaults to False.
            save_to_path (Optional[str], optional): If provided, save the dataframe to the specified path. Should end with '.csv'. Defaults to None.
            
        Raises:
            Exception: If the specified path to save the results does not work.
            Exception: If fit() was not conducted with compute_wrt_observation= True and pred_based= True.

        Returns:
            Union[pd.Series, pd.DataFrame]: Either a pd.Series with LOCO OE scores per train observation (if global) 
            or a pd.DataFrame of LOCO OE scores with train observations as columns and test observations as rows.
        """
        try:
            if local:
                if save_to_path is not None:
                    try:
                        if not os.path.exists(os.path.dirname(save_to_path)):
                            os.makedirs(os.path.dirname(save_to_path))
                        self.OE_local.to_csv(save_to_path)
                    except:
                        raise ValueError("The specified path does not work. The path should end with '.csv'.") 
                return self.OE_local
            else:
                if save_to_path is not None:
                    try:
                        if not os.path.exists(os.path.dirname(save_to_path)):
                            os.makedirs(os.path.dirname(save_to_path))
                        self.OE_global.to_csv(save_to_path)
                    except:
                        raise ValueError("The specified path does not work. The path should end with '.csv'.") 
                return self.OE_global
        except:
            raise Exception("OE values are not available. Refit with compute_wrt_observation= True and pred_based= False.")

    def heatmap(self, 
                plot_wrt_observation: bool= False,
                plot_pred_based: bool= False
                ):
        """
        Plots a heatmap of (up to 8) local LOCO values and the global LOCO values.

        LOCO scores estimate the effect on predictions or predictive performance for each feature or observation.
        When plotting with respect to predictions (instead of importance), values are always positive and their intensities indicate their effects.
        Otherwise, values may be negative which indicates that a feature is detrimental to the performance.
        
        Args:
            plot_wrt_observation (bool, optional): Whether to plot the effects/importance of observations instead of features. Defaults to False.
            plot_pred_based (bool, optional): Whether to estimate feature/observation effects instead of importances. Defaults to False.

        Raises:
            Exception: If the queried configuration has not been fit.

        CAUTION: May lead to numerical errors for small datasets (hennce not tested in pytests).
        """
        ylabel= "Test Observation"

        try:
            if plot_wrt_observation:
                xlabel= "Training Observation"
                if not plot_pred_based:
                    colorbar_label= "Loss LOCO (OI)"
                    local_df= self.OI_local
                    global_df= self.OI_global
                else:
                    colorbar_label= "Prediction LOCO (OE)"
                    local_df= self.OE_local
                    global_df= self.OE_global
                plot_title= "Local and global " + colorbar_label + " per observation"
            else:
                xlabel= "Feature"
                if not plot_pred_based:
                    colorbar_label= "Loss LOCO (FI)"
                    local_df= self.FI_local
                    global_df= self.FI_global
                else:
                    colorbar_label= "Prediction LOCO (FE)"
                    local_df= self.FE_local
                    global_df= self.FE_global
                plot_title= "Local and global " + colorbar_label + " per feature"

            plot_cols= [col for col in local_df.columns if col[:4]== "LOCO"]

            plot_data= local_df[plot_cols]
            min_LOCO= plot_data.min().min()
            max_LOCO= plot_data.max().max()
            vmax= np.max((np.abs(min_LOCO), max_LOCO))
            if not plot_pred_based:
                vmin= -vmax
            else:
                if min_LOCO < 0:
                    raise Exception('LOCO values should not be negative for feature effects.') 
                vmin= 0

            if plot_data.shape[0] > 8 or plot_data.shape[1] > 32:
                print("Pruned dataframe to ensure that it can be plotted properly.")
                plot_data= plot_data.iloc[:8, :32]

            plot_data= plot_data.append(global_df, ignore_index=True).rename(index={plot_data.shape[0]: 'Global'})
            plot_data.columns= [col[8:] for col in plot_data.columns]

            ax = sns.heatmap(plot_data, 
                             vmin= vmin, 
                             vmax= vmax, 
                             center= 0, 
                             cmap=sns.diverging_palette(10, 220, as_cmap=True, sep=50))

            ax.collections[0].colorbar.set_label(colorbar_label)
            plt.title(plot_title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

            plt.xticks(rotation= 90)
            plt.gcf().set_size_inches(8, 6)  # Adjust the width and height as needed
            plt.subplots_adjust(bottom=0.5)  # Adjust the bottom margin so feature-labels are not cut off

            plt.show()

        except:
            raise Exception("The queried configuration has not been fit. Please refit correspondingly.")


    def boxplot(self, 
                plot_wrt_observation: bool= False,
                plot_pred_based: bool= False
                ):
        """
        Plots a boxplot of local LOCO values.
        The boxplot are sorted by their overall LOCO scores/ mean values.

        LOCO scores estimate the effect on predictions or predictive performance for each feature or observation.
        When plotting with respect to predictions (instead of importance), values are always positive and their intensities indicate their effects.
        Otherwise, values may be negative which indicates that a feature is detrimental to the performance.

        Args:
            plot_wrt_observation (bool, optional): Whether to plot the effects/importance of observations instead of features. Defaults to False.
            plot_pred_based (bool, optional): Whether to estimate feature/observation effects instead of importances. Defaults to False.

        Raises:
            Exception: If the queried configuration has not been fit.
        """
        try:
            if plot_wrt_observation:
                xlabel= "Training Observation"
                if not plot_pred_based:
                    ylabel= "Loss LOCO (OI)"
                    local_df= self.OI_local
                else:
                    ylabel= "Prediction LOCO (OE)"
                    local_df= self.OE_local
                plot_title= ylabel + " per training observation"

            else:
                xlabel= "Feature"
                if not plot_pred_based:
                    ylabel= "Loss LOCO (FI)"
                    local_df= self.FI_local
                else:
                    ylabel= "Prediction LOCO (FE)"
                    local_df= self.FE_local
                plot_title= ylabel + " per feature"

            plot_cols= [col for col in local_df.columns if col[:4]== "LOCO"]

            plot_data= local_df[plot_cols]
            min_LOCO= plot_data.min().min()
            max_LOCO= plot_data.max().max()
            vmax= np.max((np.abs(min_LOCO), max_LOCO))

            plot_data.columns= [col[8:] for col in plot_data.columns]

            if plot_data.shape[1] > 32:
                print("Pruned dataframe to ensure that it can be plotted properly.")
                plot_data= plot_data.iloc[:, :32]

            plot_data_melt= pd.melt(plot_data)
            means = plot_data_melt.groupby('variable')['value'].mean()
            plot_data_melt = plot_data_melt.merge(means, left_on='variable', right_index=True, suffixes= ('', '_mean'))
            plot_data_melt = plot_data_melt.sort_values(by= "value_mean")

            # sort means in ascending order and assign a color palette
            if not plot_pred_based:
                palette = sns.diverging_palette(10, 220, n=len(means), sep= 50)
                colors = pd.Series(palette, index= means.sort_values(ascending=True).index)
            else:
                palette = sns.color_palette("ch:s=.25,rot=-.25", n_colors=len(means))
                colors = pd.Series(palette, index= means.sort_values(ascending=False).index)

            # create box plot with colors assigned to categories based on mean values
            ax = sns.boxplot(x='variable', 
                             y='value', 
                             data=plot_data_melt, 
                             palette=dict(colors),
                             showfliers= False)

            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(plot_title)

            plt.xticks(rotation= 90)
            plt.gcf().set_size_inches(8, 6)  # Adjust the width and height as needed
            plt.subplots_adjust(bottom=0.5)  # Adjust the bottom margin so feature-labels are not cut off

            plt.show()

        except:
            raise Exception("The queried configuration has not been fit. Please refit correspondingly.")
        

    def plot_bar(self, 
                plot_wrt_observation: bool= False,
                plot_pred_based: bool= False):
        """
        Plots global LOCO values as a barplot. x-axis corresponds to features or observations and the 
        y-axis describes the associated global LOCO values for predictions or predictive performance.

        Args:
            plot_wrt_observation (bool, optional): Whether to plot the effects/importance of observations instead of features. Defaults to False.
            plot_pred_based (bool, optional): Whether to estimate feature/observation effects instead of importances. Defaults to False.

        Raises:
            Exception:  If the queried configuration has not been fit.
        """

        try:
            if plot_pred_based:
                if plot_wrt_observation:
                    x_label = "Observation"
                    df = self.OE_global
                else:
                    x_label = "Feature"
                    df = self.FE_global

                palette = sns.color_palette(
                    "ch:s=.25,rot=-.25", n_colors=df.shape[0])
                plot_title = "Prediction based LOCO"
                y_label = "LOCO"
  
            else:
                if plot_wrt_observation:
                    x_label = "Observation"
                    df = self.OI_global
                else:
                    x_label = "Feature"
                    df = self.FI_global

                palette = sns.diverging_palette(10, 220, sep=3, n=df.shape[0])
                plot_title = "Loss based LOCO"
                y_label = "LOCO"

            df = pd.DataFrame(df.sort_values(ascending=True)).transpose()
            df.columns= [col[8:] for col in df.columns]

            #df = df[df.sum().sort_values(ascending=False).index]

            ax = sns.barplot(data=df, palette=palette)
            plt.title(plot_title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.xticks(rotation=90)
            plt.gcf().set_size_inches(8, 6)  # Adjust the width and height as needed
            # Adjust the bottom margin so feature-labels are not cut off
            plt.subplots_adjust(bottom=0.25)

            plt.show()

        except:
            raise Exception(
                "Barplot is only possible if fit() has been applied in advance.")
        
# from datasets.datasets import GLADdata
# loco= LOCO(data=GLADdata(id=5), 
#            debug= True,
#            n_train= 1024,
#            n_test= 64,
#            N_ensemble_configurations= 8)
# loco.fit(compute_wrt_feature=True, 
#          compute_wrt_observation=True, #
#          loss_based= True, 
#          pred_based= True)
# loco.plot_bar(plot_wrt_observation= False, plot_pred_based= False)
# loco.plot_bar(plot_wrt_observation= True, plot_pred_based= False)
# loco.plot_bar(plot_wrt_observation= False, plot_pred_based= True)
# loco.plot_bar(plot_wrt_observation= True, plot_pred_based= True)