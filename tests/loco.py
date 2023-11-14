import pytest
import itertools
from iml.methods.loco import LOCO
import matplotlib

# Test LOCO
# Execute via: python -m pytest iml/tests/loco.py

# Define ranges for each parameter
# Init-function  parameters
init_values_data = [["", ""]] #TO BE SPECIFIED
init_values_debug = [True]

# Fit-function parameters
fit_values_compute_wrt_feature = [True]
fit_values_compute_wrt_observation = [True]
fit_values_loss_based = [True]
fit_values_pred_based = [True]
fit_values_class_to_be_explained = [0, 1]

# Get-function Parameters
get_values_local = [True, False]
get_values_save_to_path = [
    None, "save_tests/loco.csv"]

# Avoid pop-up windows for plots
matplotlib.use('Agg')

# Use itertools to generate all combinations of parameters
configurations_init_fit = list(itertools.product(init_values_data,
                                        init_values_debug,
                                        fit_values_compute_wrt_feature,
                                        fit_values_compute_wrt_observation,
                                        fit_values_loss_based,
                                        fit_values_pred_based,
                                        fit_values_class_to_be_explained))

configurations_get = list(itertools.product(get_values_local,
                                            get_values_save_to_path))

# Test Function
@pytest.mark.parametrize('config', configurations_init_fit)
def test_your_function(config):
    try:
        temp_sens = LOCO(data=config[0],
                                debug=config[1])
        temp_sens.fit(compute_wrt_feature=config[2],
                      compute_wrt_observation=config[3],
                      loss_based=config[4],
                      pred_based=config[5],
                      class_to_be_explained=config[6])

        #Test FI
        for config_get in configurations_get:
            temp_res = temp_sens.get_FI(local=config_get[0],
                                        save_to_path=config_get[1])
        # temp_sens.heatmap(plot_wrt_observation=False,
        #                   plot_pred_based=False)
        temp_sens.boxplot(plot_wrt_observation=False,
                            plot_pred_based=False)

        #Test FE
        for config_get in configurations_get:
            temp_res = temp_sens.get_FE(local=config_get[0],
                                        save_to_path=config_get[1])
        # temp_sens.heatmap(plot_wrt_observation=False,
        #                   plot_pred_based=True)
        temp_sens.boxplot(plot_wrt_observation=False,
                            plot_pred_based=True)

        #Test OI
        for config_get in configurations_get:
            temp_res = temp_sens.get_OI(local=config_get[0],
                                        save_to_path=config_get[1])
        # temp_sens.heatmap(plot_wrt_observation=True,
        #                   plot_pred_based=False)
        temp_sens.boxplot(plot_wrt_observation=True,
                            plot_pred_based=False)

        #Test OE
        for config_get in configurations_get:
            temp_res = temp_sens.get_OE(local=config_get[0],
                                        save_to_path=config_get[1])
        # temp_sens.heatmap(plot_wrt_observation=True,
        #                   plot_pred_based=True)
        temp_sens.boxplot(plot_wrt_observation=True,
                            plot_pred_based=True)

    except Exception as e:
        pytest.fail(
            f"Your function raised an exception with config {config}: {str(e)}")