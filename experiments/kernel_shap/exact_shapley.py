import pandas as pd
import numpy as np
from itertools import permutations

from tabpfniml.datasets.datasets import dataset_iml
from tabpfniml.methods.interpret import TabPFN_Interpret

class Shapley_Exact(TabPFN_Interpret):
    """
        ...
    """
    def __init__(self,
                 data: dataset_iml,
                 n_train: int = 1024,
                 n_test: int = 512,
                 N_ensemble_configurations: int = 16,
                 device: str = "cpu",
                 debug: bool = False):
        """
            ...
            #effort scales in 2^p
        """
        super().__init__(data=data,
                         n_train=n_train,
                         n_test=n_test,
                         N_ensemble_configurations= N_ensemble_configurations,
                         device=device,
                         debug=debug,
                         standardize_features=False,
                         to_torch_tensor=False,
                         store_gradients=False)

    def fit(self,
            class_to_be_explained: int = 1,
            debug= False
            ):
        self.class_to_be_explained= class_to_be_explained

        #Generate all p! feature permutations
        feature_indices_list = list(range(self.data.num_features))
        feature_permutations = list(permutations(feature_indices_list))

        #Initialize dataframe to track marginal contributions of features
        marg_cont_df= pd.DataFrame(columns= [j for j in range(self.data.num_features)])

        #Set relative frequency of class 1 as mean pred
        self.mean_pred= self.y_train.mean()

        if debug:
            feature_permutations= feature_permutations[:32]

        for permutation in feature_permutations:
            temp_marg_cont= {}
            prev_pred_in_coalition= np.full(self.X_test.shape[0], self.mean_pred)
            
            #For each permutation and for all first k elements for k=1,...,p refit the model and compute
            #the difference in prediction to the subset of the first k-1 elements
            for permutation_index in range(len(permutation)):
                temp_features= permutation[:permutation_index+1]

                X_train_masked = self.X_train[:, temp_features].copy()
                X_test_masked = self.X_test[:, temp_features].copy()

                self.classifier.fit(X_train_masked, self.y_train.copy())
                temp_pred= self.classifier.predict_proba(X_test_masked)[:, self.class_to_be_explained] #, return_logits= True

                temp_marg_cont[temp_features[-1]]= list(temp_pred - prev_pred_in_coalition)
                prev_pred_in_coalition= temp_pred

            marg_cont_df= marg_cont_df.append(temp_marg_cont, ignore_index=True)
        
        marg_cont_array= np.array(marg_cont_df.to_numpy().tolist())
        self.shapley_values= np.transpose(marg_cont_array.mean(axis=0))