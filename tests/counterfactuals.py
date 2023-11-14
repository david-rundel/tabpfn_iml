import pytest
import itertools
from iml.methods.counterfactuals import Counterfactuals
import matplotlib

# Test CE
# Execute via: python -m pytest iml/tests/counterfactuals.py

# Define ranges for each parameter
# Init-function  parameters
init_values_data = [["", ""]] #TO BE SPECIFIED
init_values_debug = [False] #Do not debug in test mode to enable initialization based on test set with sufficiently big dataset.

# Fit-function parameters
fit_values_test_index_factual = [0, 1]
fit_values_non_actionable_features = [None, "STR_RANDOM_NAFs", [0, 1, 2]]
fit_values_actionable_features = [None, "STR_RANDOM_AFs", [0, 1, 2, 3, 4]]
fit_values_init_based_on_test_set = [True, False]
#fit_values_init_poisson_lambda = [1, 3]
fit_values_print_time = [True]

# Get-function Parameters
get_first_n = [10]
get_values_save_to_path = [None, "save_tests/ce.csv"]

# Avoid pop-up windows for plots
matplotlib.use('Agg')

# Use itertools to generate all combinations of parameters
configurations_init_fit = list(itertools.product(init_values_data,
                                        init_values_debug,
                                        fit_values_test_index_factual,
                                        fit_values_non_actionable_features,
                                        fit_values_actionable_features,
                                        fit_values_init_based_on_test_set,
                                        fit_values_print_time))

configurations_get = list(itertools.product(get_first_n,
                                            get_values_save_to_path))

# Test Function
@pytest.mark.parametrize('config', configurations_init_fit)
def test_your_function(config):
    try:
        temp_ce = Counterfactuals(data=config[0],
                                debug=config[1], 
                                n_train= 32,
                                n_test= 256,
                                N_ensemble_configurations= 2)

        #Filter configurations
        if config[3] != None and config[4] != None:
            pass

        if config[3] != None and config[5]:
            pass

        if config[4] != None and config[5]:
            pass

        if config[3] == "STR_RANDOM_NAFs":
            temp_nafs = list(temp_ce.data.feature_names[0: 3])
        else:
            temp_nafs = config[3]

        if config[4] == "STR_RANDOM_AFs":
            temp_afs = list(temp_ce.data.feature_names[0: 5])
        else:
            temp_afs = config[4]

        temp_ce.fit(test_index_factual=config[2],
                    non_actionable_features= temp_nafs,
                    actionable_features= temp_afs,
                    init_based_on_test_set= config[5],
                    print_time=config[6],
                    population_size= 15,
                    offsprings_size= 5,
                    n_iter= 3)

        for config_get in configurations_get:
            temp_ce.get_counterfactuals(first_n=config_get[0],
                                        save_to_path=config_get[1])

    except Exception as e:
        pytest.fail(
            f"Your function raised an exception with config {config}: {str(e)}")