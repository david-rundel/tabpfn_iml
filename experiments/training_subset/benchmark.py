import sys
sys.path.append('/Users/davidrundel/git/tabpfn_iml/')

import numpy as np
import pandas as pd
from scipy.special import softmax
import random
import math
import torch
from sklearn.metrics import accuracy_score

from tabpfniml.datasets.datasets import OpenMLData
from tabpfniml.methods.sensitivity import Sensitivity

"""
toGPT:
Script to find an optimal training subset of a dataset which does not fit into a single TabPFN forward pass
exploiting information gained from Sensitivity Analysis for Observation Importance (data valuation).

In detail, we divide the dataset into a train-, val- and test split. The train split is further
divided into chunks that can be processes in a single forward pass (max_forward_pass). For each chunk,
the chunk is used as TabPFN training set and the val-split as inference sample. Using Sensitivity Analysis
for data valuation, the gradients of the val-split with respect to the training set are computed and 
taken as data values/ relevance scores. We repeat this for several runs with various, randomly sampled training split chunks 
and. In the end we choose the optimal training split subset of a size that
can be fit in a single TabPFN forward pass based on the average relevance scores. The test-split is then used to get an unbiased estimate of the optimal training set.
"""
#exc: if not too much samples for single forward pass
#del if mean 0 or 4 statement


experiment_runs= 5

random.seed(42)
experiment_seeds= [random.randint(1, 10000) for _ in range(experiment_runs)]

def optimize_training_set(openml_id: int = 819,
                          experiment_seed: int = 728,
                          max_forward_pass: int= 256, #Specify the amount of data that can be processed in TabPFN forward passes (based on the utilized harware, this can also be lower than 1024)
                          multiple_of_fp: int= 5, #Multiple of forward pass size to consider
                          val_set_size= 128, 
                          test_set_size=128,
                          internal_runs= 10,
                          compare_accuracy= True
                          ):

    #Initialize Sensitivity with (too big) training dataset
    data= OpenMLData(openml_id, avoid_pruning= True, seed= experiment_seed)

    #Specify the size of the subset of the entire dataset to be optimized 
    ds_opt_size = int(min(multiple_of_fp * max_forward_pass, data.num_samples))

    if data.num_samples <= max_forward_pass:
        raise ValueError("There is no need to specify an optimal subset, since the entire dataset fits in a single TabPFN  forward pass.")

    #Corrupt maximum amount of samples so that entire dataset is stored in object
    data.max_n_train= data.num_samples

    sensitivity= Sensitivity(data= data,
                            n_train= min(data.num_samples - test_set_size, ds_opt_size + val_set_size),
                            n_test= test_set_size,
                            N_ensemble_configurations=16)

    X_train= sensitivity.X_train[:ds_opt_size, :].copy()
    y_train= sensitivity.y_train[:ds_opt_size].copy()

    X_val= sensitivity.X_train[ds_opt_size: ds_opt_size + val_set_size, :].copy()
    y_val= sensitivity.y_train[ds_opt_size: ds_opt_size + val_set_size].copy()

    X_test= sensitivity.X_test.copy()
    y_test= sensitivity.y_test.copy()

    X_train_tensor = torch.tensor(X_train.copy(), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.copy(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.copy(), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.copy()).type(torch.LongTensor)

    del sensitivity.X_train
    del sensitivity.y_train
    del sensitivity.X_test
    del sensitivity.y_test
    del sensitivity.n_train
    del sensitivity.n_test

    train_indices= list(range(X_train.shape[0]))

    sens_scores= np.zeros((len(train_indices), internal_runs))

    for run in range(internal_runs):
        #For each run divide the train test into several chunks
        run_train_indices= train_indices.copy()
        random.shuffle(run_train_indices)

        amount_chunks= math.ceil(len(run_train_indices)/max_forward_pass)
        run_train_chunks= [run_train_indices[i * max_forward_pass: (i+1) * max_forward_pass] for i in range(amount_chunks)]
        
        #For each chunk obtain scores on validation set
        for train_chunk in run_train_chunks:
            sensitivity.X_train= X_train[train_chunk,:]
            sensitivity.y_train= y_train[train_chunk]
            sensitivity.X_test= X_val
            sensitivity.y_test= y_val

            sensitivity.n_train= sensitivity.X_train.shape[0]
            sensitivity.n_test= sensitivity.X_test.shape[0]

            sensitivity.fit(compute_wrt_feature= False,
                            compute_wrt_observation= True,
                            pred_based= False,
                            loss_based= True,
                            comp_global = True)

            sens_scores[train_chunk ,run]= sensitivity.OI_global.values #softmax() #wegen softmax immer 0.00

    best_train_samples_1= np.argsort(sens_scores.mean(axis=1))[:max_forward_pass]
    best_train_samples_2= np.argsort(sens_scores.mean(axis=1))[-max_forward_pass:] #even worse (highest sens)

    if compare_accuracy:
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        #Compare random subset against optimized subset
        sensitivity.classifier.fit(X_train_tensor[:max_forward_pass,:], 
                                y_train_tensor[:max_forward_pass])
        preds_random_train = sensitivity.classifier.predict_proba(X_test_tensor)
        loss_random_train = criterion(preds_random_train, y_test_tensor)  # .detach().numpy()
        # preds_random_train_hard = sensitivity.classifier.classes_.take(np.asarray(np.argmax(preds_random_train, axis=-1), dtype=np.intp))
        # acc_random_train= accuracy_score(y_test_tensor, preds_random_train_hard)

        sensitivity.classifier.fit(X_train_tensor[best_train_samples_1,:], 
                                y_train_tensor[best_train_samples_1])
        preds_opt_train_1 = sensitivity.classifier.predict_proba(X_test_tensor)
        loss_random_train_1 = criterion(preds_opt_train_1, y_test_tensor)  # .detach().numpy()
        # preds_opt_train_hard = sensitivity.classifier.classes_.take(np.asarray(np.argmax(preds_opt_train, axis=-1), dtype=np.intp))
        # acc_opt_train= accuracy_score(y_test_tensor, preds_opt_train_hard)

        sensitivity.classifier.fit(X_train_tensor[best_train_samples_2,:], 
                                y_train_tensor[best_train_samples_2])
        preds_opt_train_2 = sensitivity.classifier.predict_proba(X_test_tensor)
        loss_random_train_2 = criterion(preds_opt_train_2, y_test_tensor)  # .detach().numpy()

    return best_train_samples_1, {"acc_random_train_set": loss_random_train, "acc_optimized_train_set_1": loss_random_train_1, "acc_optimized_train_set_2": loss_random_train_2}, sens_scores

experiment_results= []
sens_scores= []

for experiment_run in range(experiment_runs):
    best_train_samples, acc_dict, temp_sens_scores= optimize_training_set(openml_id= 819, experiment_seed= experiment_seeds[experiment_run])
    experiment_results.append(acc_dict)
    sens_scores.append(temp_sens_scores)

print(experiment_results)