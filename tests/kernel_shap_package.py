import pytest
import itertools
from iml.methods.kernel_shap_package import SHAP_Package_Wrapper
import matplotlib

# Test Kernel SHAP Package Wrapper
# Execute via: python -m pytest iml/tests/kernel_shap_package.py

# Define ranges for each parameter
# Init-function  parameters
init_values_data = [["", ""]] #TO BE SPECIFIED
init_values_debug = [True]

# Fit-function parameters
fit_values_approximate_marginalization = [True, False]
fit_values_log_odds_units = [True, False]
fit_values_class_to_be_explained = [0, 1]

# Get-function Parameters
get_values_local = [True, False]
get_values_save_to_path = [None, "save_tests/shap_package.csv"]

# Plot-function Parameters
plot_values_test_index = [0, 1]
plot_values_dependent_feature = [0, "FEAT_NAME_STR"]

# Avoid pop-up windows for plots
matplotlib.use('Agg')

# Use itertools to generate all combinations of parameters
configurations_init_fit = list(itertools.product(init_values_data,
                                        init_values_debug,
                                        fit_values_approximate_marginalization,
                                        fit_values_log_odds_units,
                                        fit_values_class_to_be_explained))

configurations_get = list(itertools.product(get_values_local,
                                            get_values_save_to_path))
configurations_plot = list(itertools.product(plot_values_test_index,
                                             plot_values_dependent_feature))

# Test Function
@pytest.mark.parametrize('config', configurations_init_fit)
def test_your_function(config):
    try:
        temp_shap = SHAP_Package_Wrapper(data=config[0],
                                         debug=config[1])
        temp_shap.fit(approximate_marginalization=config[2],
                      log_odds_units=config[3],
                      class_to_be_explained=config[4])

        for config_get in configurations_get:
            temp_shap.get_SHAP_values(local=config_get[0],
                                      save_to_path=config_get[1])
            
        for config_plot in configurations_plot:
            temp_shap.plot_force(test_index=config_plot[0])
            temp_shap.plot_summary()

            if config_plot[1] == "FEAT_NAME_STR":
                # One of the features specified as string
                feature_name_str = temp_shap.data.feature_names[0]
                temp_shap.plot_dependence(dependent_feature=feature_name_str)
            else:
                temp_shap.plot_dependence(dependent_feature=config_plot[1])

    except Exception as e:
        pytest.fail(
            f"Your function raised an exception with config {config}: {str(e)}")