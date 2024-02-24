from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Optional, Union, Tuple
import torch
from pathlib import Path
import os
import openml

class dataset_iml(ABC):
    """
    Abstract parent-class for all datasets used in the analysis of TabPFN for biomechanics.
    From information provided in each of the sub-classes __init__-method, it infers all relevant attributes of a dataset 
    needed for this repository and enables loading the dataset with various specifications.
            
    """
    project_dir = os.getcwd()

    def __init__(self):
        """
        Initialize the corresponding dataset object, load the file and infer relevant metadata.

        Requires the subclass to specify self.id, self.path, self.id_to_file, self.id_to_target_name, self.categorical_features 
        in the __init__()-method and then call the parents __init__()-method.

        TODO:
            -Track nominal and ordinal features.
        """
        #Ensure that relevant attributes were set in the subclasses __init__()-method
        assert hasattr(self, "id"), "The 'id' attribute has to be set in the subclasses __init__()-method."
        assert hasattr(self, "path"), "The 'path' attribute has to be set in the subclasses __init__()-method."
        assert hasattr(self, "id_to_file"), "The 'id_to_file attribute' has to be set in the subclasses __init__()-method."
        assert hasattr(self, "id_to_target_name"), "The 'id_to_target_name' attribute has to be set in the subclasses __init__()-method."
        assert hasattr(self, "categorical_features"), "The 'categorical_features' attribute has to be set in the subclasses __init__()-method."

        self.path_dataset = str(dataset_iml.project_dir) + "/" + self.path + "/" + self.id_to_file[self.id]

        #Load the dataset as df
        self.df= pd.read_csv(self.path_dataset)
        self.X_df = self.df.iloc[:, 1:]
        self.y_df = self.df.iloc[:, 0]

        #Convert to np-array
        self.Xy = self.df.to_numpy()
        self.X = self.Xy[:,1:]
        self.y = self.Xy[:,0]

        # infer relevant metadata
        self.y_classes = len(np.unique(self.y))
        self.feature_names= self.df.columns[1:]
        self.num_features= self.X.shape[1]
        self.num_samples= self.X.shape[0]

        self.levels_per_feature= np.apply_along_axis(lambda x: len(np.unique(x)), axis=0, arr=self.X)
        self.categorical_features_idx = [self.feature_name_to_index(feature_name) for feature_name in self.categorical_features]
        self.continuous_features = list(set(self.feature_names)-set(self.categorical_features))
        self.continuous_features_idx = [self.feature_name_to_index(feature_name) for feature_name in self.continuous_features]

        self.max_n_train= 1024

        #TODO: Track nominal and ordinal features.

    def load(self,
             split: bool= True,
             n_train: Optional[int]= None,
             n_test: Optional[int]= None,
             standardize_features_bool: bool= False,
             to_torch_tensor: bool= False, #y_test_desired: Optional[int]= None,
             device: str= "cpu") -> Union[Tuple[np.ndarray, np.ndarray], 
                                                    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                                                    Tuple[torch.tensor, torch.tensor],
                                                    Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]]:
        """
        Load the dataset with specified preprocessing.

        Args:
            split (bool, optional): Whether to split the data into train- and test-sets. Defaults to True.
            n_train (Optional[int], optional): The amount of samples in the train-set. Defaults to None.
            n_test (Optional[int], optional): The amount of samples in the test-set. Defaults to None.
            standardize_features (bool, optional): Whether to standardize each feature. Defaults to False.
            to_torch_tensor (bool, optional): Whether the data should be returned as torch.tensor instead of np.ndarray.
            device (str, optional): The device to store tensors and the TabPFN model on. Defaults to "cpu".
            
        Returns:
            Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[torch.tensor, torch.tensor], Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]]: 
            Arrays for the independent and dependent variables split into train and test samples (X_train, X_test, y_train, y_test) or unsplit (X, y).
        """

        def standardize_features(X: np.ndarray, 
                                 X_test: Optional[np.ndarray]= None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
            """
            Standardizes the features of a np.ndarray. If two arrays are provided, they are interpreted as train- and test-set
            and the statistics for standardization are solely derived on the train-set.

            Args:
                X (np.ndarray): The data to be standardized or, if X_test is provided, the train-set to be standardized.
                X_test (Optional[np.ndarray], optional): The test-set to be standardized. Defaults to None.

            Returns:
                Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: The standardized data or standardized train- and test-set.
            """
            mean= X.mean(axis=0)
            std= X.std(axis=0)
            eps= 0.000001

            if isinstance(X_test, np.ndarray):
                return ((X-mean)/(std+eps)), ((X_test-mean)/(std+eps))
            else:
                return ((X-mean)/(std+eps))
            
        X= self.X
        y= self.y
    
        if not split:
            if standardize_features_bool:
                X= standardize_features(X)
            if to_torch_tensor:
                X= torch.tensor(X, device= device)
                y= torch.tensor(y, device= device).type(torch.LongTensor)
            return X, y

        else:
            if n_train:
                if (n_train <= self.max_n_train) and (n_train < self.num_samples):
                    pass
                else:
                    n_train= min(self.max_n_train, int(self.num_samples * 0.66))
            else:
                if (self.num_samples - self.max_n_train) > 0:
                    n_train= self.max_n_train
                else:
                    n_train= int(self.num_samples * 0.66)
                    # TODO Print warning

            if n_test:
                if (self.num_samples - n_train - n_test) >= 0:
                    pass
                else:
                    n_test= min(n_test, self.num_samples - n_train)
                    # TODO Print warning

            else:
                n_test= (self.num_samples - n_train)

            X_train, X_test, y_train, y_test= train_test_split(X,
                                                               y,
                                                               train_size= n_train,
                                                               stratify= y, 
                                                               random_state= 42)
            
            #Maintain  raw (not standardized, transformed or reduced train-test-split)
            #Used e.g. for counterfactuals to ensure that enough observations n test-set with desired prediction are available
            #self.X_train_raw, self.X_test_raw, self.y_train_raw, self.y_test_raw= X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()
            
            self.X_train_df, self.X_test_df, self.y_train_df, self.y_test_df= train_test_split(self.X_df,
                                                               self.y_df,
                                                               train_size= n_train,
                                                               stratify= y, 
                                                               random_state= 42)
            
            if standardize_features_bool:
                #Standardize before subsetting to maintain meaningful statistics
                X_train, X_test= standardize_features(X_train, X_test)

            if n_train and n_test:
                X_test = X_test[:n_test,:]
                y_test = y_test[:n_test]
                self.X_test_df = self.X_test_df.iloc[:n_test,:]
                self.y_test_df = self.y_test_df.iloc[:n_test]

            if to_torch_tensor:        
                X_train= torch.tensor(X_train, device= device)
                X_test= torch.tensor(X_test, device= device)
                y_train= torch.tensor(y_train, device= device)
                y_test= torch.tensor(y_test, device= device).type(torch.LongTensor)

            # TODO: Check whether to compute min and max vals on train or test data?
            if not to_torch_tensor:
                self.min_vals = np.min(X_test, axis=0)
                self.max_vals = np.max(X_test, axis=0)

            return X_train, X_test, y_train, y_test
    
    
    def feature_name_to_index(self,
                              feature_name: str) -> int:
        """
        Returns the feature index given the feature name.

        Args:
            feature_name (str): Name of the feature in the dataset.

        Raises:
            Exception: If the specified feature name does not occur in the dataset.

        Returns:
            int: Index of the feature in the dataset.
        """
        try:
            return int(list(self.feature_names).index(feature_name))
        except:
            raise Exception("The specified feature name does not occur in the dataset.")
        


class OpenMLData(dataset_iml):
    def __init__(self, 
                 openml_id: int= 1,
                 avoid_pruning: bool= False,
                 seed: int= 728):
        
        np.random.seed(seed)

        def fetch_openml_dataset(did):
            """
            Helper-Method:
            Fetches a single dataset from OpenML by its ID.

            Restricted to array-datasets.
            
            Args:
            - did: Dataset ID on OpenML.
            
            Returns:
            - A tuple containing the dataset name, features (X), target variable (y),
            indices of categorical features, and feature names.
            """
            dataset = openml.datasets.get_dataset(did)
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                target=dataset.default_target_attribute,
                dataset_format='dataframe')
            
            X= X.to_numpy()
            y= y.to_numpy()
            
            #Shuffle data (some OpenML datasets exhibit long sequences with identical label)
            shuffle_indices = np.random.permutation(len(y))
            X = X[shuffle_indices]
            y = y[shuffle_indices]
            
            if X.shape[0] > 1024 and not avoid_pruning:
                X, _, y, _ = train_test_split(X, y, train_size=1024, stratify=y, random_state= seed)
            
            # Find indices of categorical features
            categorical_features_idx = [i for i, is_categorical in enumerate(categorical_indicator) if is_categorical]

            le = LabelEncoder()
            y = le.fit_transform(y)
            
            return dataset.name, X, y, categorical_features_idx, attribute_names


        # Set attributes directly without relying on a file path
        name, X, y, categorical_features_idx, feature_names= fetch_openml_dataset(openml_id)

        self.id = 1
       
        self.df = pd.DataFrame(X, columns=feature_names)
        self.df['target'] = y 
        self.categorical_features = [feature_names[i] for i in categorical_features_idx]

        # Initialize attributes that would be inferred from a file
        self.X_df = self.df.drop(columns=['target'])
        self.y_df = self.df['target']
        
        # Convert to numpy arrays for consistency with the ConsultingData class
        self.Xy = self.df.to_numpy()
        self.X = self.X_df.to_numpy()
        self.y = self.y_df.to_numpy()

        # infer relevant metadata
        self.y_classes = len(np.unique(self.y))
        self.feature_names = np.array(feature_names)
        self.num_features = self.X.shape[1]
        self.num_samples = self.X.shape[0]

        self.feature_complete_names = {}

        self.levels_per_feature = np.apply_along_axis(lambda x: len(np.unique(x)), axis=0, arr=self.X)
        self.categorical_features_idx = categorical_features_idx
        self.continuous_features = list(set(self.feature_names[:-1]) - set(self.categorical_features))
        self.continuous_features_idx = [i for i, feature_name in enumerate(self.feature_names[:-1]) if feature_name in self.continuous_features]

        # No need to set a path for dataset since data is directly passed
        # self.path_dataset = Not needed
        
        # Set project_dir to current directory or any specific directory as a placeholder
        self.project_dir = Path.cwd()
        self.max_n_train= 1024

        # No need to call super().__init__() as we're directly initializing everything here
        # The original intent of calling super().__init__() was to load and process the file, which we've bypassed



class ArrayData(dataset_iml):
    def __init__(self, name: str, X, y, categorical_features_idx: list, feature_names: list, id_to_target_name: dict = None):
        # Set attributes directly without relying on a file path
        self.id = 1
       
        self.df = pd.DataFrame(X, columns=feature_names)
        self.df['target'] = y 
        self.categorical_features = [feature_names[i] for i in categorical_features_idx]

        # Initialize attributes that would be inferred from a file
        self.X_df = self.df.drop(columns=['target'])
        self.y_df = self.df['target']
        
        # Convert to numpy arrays for consistency with the ConsultingData class
        self.Xy = self.df.to_numpy()
        self.X = self.X_df.to_numpy()
        self.y = self.y_df.to_numpy()

        # infer relevant metadata
        self.y_classes = len(np.unique(self.y))
        self.feature_names = np.array(feature_names)
        self.num_features = self.X.shape[1]
        self.num_samples = self.X.shape[0]

        self.feature_complete_names = {}
        self.id_to_target_name = id_to_target_name

        self.levels_per_feature = np.apply_along_axis(lambda x: len(np.unique(x)), axis=0, arr=self.X)
        self.categorical_features_idx = categorical_features_idx
        self.continuous_features = list(set(self.feature_names[:-1]) - set(self.categorical_features))
        self.continuous_features_idx = [i for i, feature_name in enumerate(self.feature_names[:-1]) if feature_name in self.continuous_features]

        # No need to set a path for dataset since data is directly passed
        # self.path_dataset = Not needed
        
        # Set project_dir to current directory or any specific directory as a placeholder
        self.project_dir = Path.cwd()
        self.max_n_train= 1024