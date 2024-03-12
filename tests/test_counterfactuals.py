import os
import sys

current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_dir)
sys.path.append(parent_dir)

import matplotlib
from tabpfniml.datasets.datasets import OpenMLData
from tabpfniml.methods.counterfactuals import Counterfactuals
import itertools
import pytest


# Test CE
# Execute via: python -m pytest tests/counterfactuals.py

# Define ranges for each parameter
# Init-function  parameters
init_values_data = [770, 819]
# Do not debug in test mode to enable initialization based on test set with sufficiently big dataset.
init_values_debug = [False]

# Fit-function parameters
fit_values_test_index_factual = [0, 1]
fit_values_non_actionable_features = [None, "STR_RANDOM_NAFs", [0, 1, 2]]
fit_values_actionable_features = [None, "STR_RANDOM_AFs", [0, 1, 2, 3, 4]]
fit_values_init_based_on_test_set = [True, False]
# fit_values_init_poisson_lambda = [1, 3]
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


@pytest.mark.parametrize('config', [c for c in configurations_init_fit if not (
    (c[3] is not None and c[4] is not None) or
    (c[3] is not None and c[5]) or
    (c[4] is not None and c[5]))])
def test_your_function(config):
    try:
        temp_ce = Counterfactuals(data=OpenMLData(openml_id=config[0]),
                                  debug=config[1],
                                  n_train=32,
                                  n_test=256,
                                  N_ensemble_configurations=2)

        if config[3] == "STR_RANDOM_NAFs":
            temp_nafs = list(temp_ce.data.feature_names[0: 3])
        else:
            temp_nafs = config[3]

        if config[4] == "STR_RANDOM_AFs":
            temp_afs = list(temp_ce.data.feature_names[0: 5])
        else:
            temp_afs = config[4]

        temp_ce.fit(test_index_factual=config[2],
                    non_actionable_features=temp_nafs,
                    actionable_features=temp_afs,
                    init_based_on_test_set=config[5],
                    print_time=config[6],
                    population_size=15,
                    offsprings_size=5,
                    n_iter=3)

        for config_get in configurations_get:
            temp_ce.get_counterfactuals(first_n=config_get[0],
                                        save_to_path=config_get[1])

    except Exception as e:
        pytest.fail(
            f"Your function raised an exception with config {config}: {str(e)}")
