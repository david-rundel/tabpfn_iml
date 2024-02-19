import numpy as np
import pandas as pd
import openml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.inspection import partial_dependence
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier

from tabpfniml.datasets.datasets import ConsultingData




def fetch_openml_dataset(did):
    """
    Fetches a single dataset from OpenML by its ID.
    
    Args:
    - did: Dataset ID on OpenML.
    
    Returns:
    - A tuple containing the dataset name, features (X), target variable (y),
      indices of categorical features, and feature names.
    """
    dataset = openml.datasets.get_dataset(did)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute,
        dataset_format='array')
    
    if X.shape[0] > 1024:
        X, _, y, _ = train_test_split(X, y, train_size=1024, stratify=y, random_state=42)
    
    # Find indices of categorical features
    categorical_features_idx = [i for i, is_categorical in enumerate(categorical_indicator) if is_categorical]

    le = LabelEncoder()
    y = le.fit_transform(y)
    
    return dataset.name, X, y, categorical_features_idx, attribute_names



class OpenMLData(ConsultingData):
    def __init__(self, name, X, y, categorical_features_idx, feature_names):
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


# OpenML dataset IDs from TabPFN paper:
open_cc_dids = [11, 14, 15, 16, 18, 22, 23, 31, 37, 50] # 54, 188, 458, 469, 1049, 1050, 1063, 1068, 1510, 1494, 1480, 1462, 1464, 6332, 23381, 40966, 40982, 40994, 40975
open_cc_dids_debug = [11, 22, 31, 37]


# pima diabetes dataset
diabetes = pd.read_csv('https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv')

X = diabetes.drop(columns='Outcome')
y = diabetes['Outcome'] 

diabetes_data = OpenMLData("diabetes", X, y, categorical_features_idx=[0], feature_names=X.columns)
diabetes_data.id_to_target_name = {1: 'Diabetes'}

n_train = int(0.8 * X.shape[0])
n_test = int(0.2 * y.shape[0])
