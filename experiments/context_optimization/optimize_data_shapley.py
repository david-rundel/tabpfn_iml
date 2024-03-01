import sys
sys.path.append('/Users/davidrundel/git/tabpfn_iml/')

from tabpfniml.methods.data_shapley import Data_Shapley

debug= True
if debug:
    n_train= 1024
    n_val= 128
    n_test= 256
    M_factor= 1
    tPFN_train_min= 128
    tPFN_train_max= 256

# else:
    # n_train= 1024
    # n_val= 128
    # n_test= 256
    # M_factor= 1
    # tPFN_train_min= 128
    # tPFN_train_max= 256

openml_id= 819
seed= 728

data_shapley= Data_Shapley(optimize_context= True,
                           openml_id= openml_id,
                           n_train= n_train,
                           n_val= n_val,
                           n_test= n_test,
                           seed= seed
                           )

data_shapley.fit(M_factor= M_factor,
                 tPFN_train_min= tPFN_train_min,
                 tPFN_train_max= tPFN_train_max,
                 class_to_be_explained= 1)

data_values= data_shapley.get_data_values()
opt_context= data_shapley.get_optimized_context()
perf_diff= data_shapley.get_optimized_performance_diff("experiments/context_optimization/results/run_results.pkl")
print("Stop!")
#TO DELETE:
data_values= data_shapley.get_data_values()
opt_context= data_shapley.get_optimized_context()
perf_diff= data_shapley.get_optimized_performance_diff("experiments/context_optimization/results/run_results.pkl")