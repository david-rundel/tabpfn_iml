import os
import sys

current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_dir)
sys.path.append(parent_dir)

import matplotlib
from tabpfniml.datasets.datasets import OpenMLData
from tabpfniml.methods.ale import ALE
import itertools
import pytest


# Test ALE
# Execute via: python -m pytest tests/ale.py

# Define ranges for each parameter
# Init-function  parameters
init_values_data = [770, 819]
init_values_debug = [True]

# Fit-function parameters
fit_values_features = [None, "First_numerical"]
fit_values_max_intervals_per_feature = [5, 10]
fit_values_center = [False, True]
fit_values_discretize_by_linear_spacing = [False, True]

# Get-PD-function parameters
get_values_feature = ["First_numerical"]
get_values_save_to_path = [None, "save_tests/ale.csv"]

# Plot-function parameters
plot_values_features = [None, "First_numerical"]
plot_values_save_to_dir = [None, "save_tests"]

# Avoid pop-up windows for plots
matplotlib.use('Agg')

configurations_is_init_fit = list(itertools.product(init_values_data,
                                                    init_values_debug,
                                                    fit_values_features,
                                                    fit_values_max_intervals_per_feature,
                                                    fit_values_center,
                                                    fit_values_discretize_by_linear_spacing))
configurations_is_get = list(itertools.product(get_values_feature,
                                               get_values_save_to_path))
configurations_is_plot = list(itertools.product(plot_values_features,
                                                plot_values_save_to_dir))

numerical_feat_per_dataset = {770: ["country_code"], 819: ["Altitude"]}

# Test Function


@pytest.mark.parametrize('config', configurations_is_init_fit)
def test_your_function(config):
    try:
        temp_ale = ALE(data=OpenMLData(openml_id=config[0]),
                       debug=config[1])
        temp_ale.fit(features=config[2] if config[2] is None else numerical_feat_per_dataset[config[0]],
                     max_intervals_per_feature=config[3],
                     center=config[4],
                     discretize_by_linear_spacing=config[5])

        for config_get in configurations_is_get:
            temp_ale.get_ALE(feature=numerical_feat_per_dataset[config[0]][0],
                             save_to_path=config_get[1])

        for config_plot in configurations_is_plot:
            temp_ale.plot(features=config_plot[0] if config_plot[0] is None else numerical_feat_per_dataset[config[0]],
                          save_to_dir=config_plot[1])

    except Exception as e:
        pytest.fail(
            f"Your function raised an exception: {str(e)}")
