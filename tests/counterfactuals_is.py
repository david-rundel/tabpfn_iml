import pytest
import itertools
from iml.methods.counterfactuals import Counterfactuals
import matplotlib

#Test CE In-Sample-Search
# Execute via: python -m pytest iml/tests/counterfactuals_is.py

# Define ranges for each parameter
# Init-function  parameters
init_values_data = [["", ""]] #TO BE SPECIFIED
init_values_debug = [True]

# Fit-function parameters
fit_values_test_index_factual = [0, 1]

# Get-function Parameters
get_first_n = [10]
get_values_save_to_path = [None, "save_tests/ce_is.csv"]

# Avoid pop-up windows for plots
matplotlib.use('Agg')

configurations_is_init_fit = list(itertools.product(init_values_data,
                                                    init_values_debug,
                                                    fit_values_test_index_factual))   
configurations_is_get = list(itertools.product(get_first_n,
                                               get_values_save_to_path))

# Test Function
@pytest.mark.parametrize('config', configurations_is_init_fit)
def test_your_function(config):
    try:
        temp_ce = Counterfactuals(data=config[0],
                                  debug=config[1])
        temp_ce.fit(test_index_factual= config[2],
                    in_sample_search= True)

        for config_get in configurations_is_get:
            temp_ce.get_counterfactuals(first_n=config_get[0],
                                        save_to_path=config_get[1])

    except Exception as e:
        pytest.fail(
            f"Your function raised an exception with config {config}: {str(e)}")