import pytest
import itertools
from iml.methods.sensitivity import Sensitivity
import matplotlib

# Test Sensitivity
# Execute via: python -m pytest iml/tests/sensitivity.py

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
    None, "save_tests/sensitivity_package.csv"]

# Plot-force-function Parameters
plot_values_bins = [10, 20]
plot_values_log_scale = [[False, True], [False, False]]

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

configurations_plot = list(itertools.product(plot_values_bins,
                                             plot_values_log_scale))

# Test Function
@pytest.mark.parametrize('config', configurations_init_fit)
def test_your_function(config):
    try:
        temp_sens = Sensitivity(data=config[0],
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
        for config_plot in configurations_plot:
            temp_sens.plot_histogram(plot_wrt_observation=False,
                                        plot_pred_based=False,
                                        bins=config_plot[0],
                                        log_scale=config_plot[1])

        #Test FE
        for config_get in configurations_get:
            temp_res = temp_sens.get_FE(local=config_get[0],
                                        save_to_path=config_get[1])
        # temp_sens.heatmap(plot_wrt_observation=False,
        #                   plot_pred_based=True)
        temp_sens.boxplot(plot_wrt_observation=False,
                            plot_pred_based=True)
        for config_plot in configurations_plot:
            temp_sens.plot_histogram(plot_wrt_observation=False,
                                        plot_pred_based=True,
                                        bins=config_plot[0],
                                        log_scale=config_plot[1])

        #Test OI
        for config_get in configurations_get:
            temp_res = temp_sens.get_OI(local=config_get[0],
                                        save_to_path=config_get[1])
        # temp_sens.heatmap(plot_wrt_observation=True,
        #                   plot_pred_based=False)
        temp_sens.boxplot(plot_wrt_observation=True,
                            plot_pred_based=False)
        for config_plot in configurations_plot:
            temp_sens.plot_histogram(plot_wrt_observation=True,
                                        plot_pred_based=False,
                                        bins=config_plot[0],
                                        log_scale=config_plot[1])

        #Test OE
        for config_get in configurations_get:
            temp_res = temp_sens.get_OE(local=config_get[0],
                                        save_to_path=config_get[1])
        # temp_sens.heatmap(plot_wrt_observation=True,
        #                   plot_pred_based=True)
        temp_sens.boxplot(plot_wrt_observation=True,
                            plot_pred_based=True)
        for config_plot in configurations_plot:
            temp_sens.plot_histogram(plot_wrt_observation=True,
                                        plot_pred_based=True,
                                        bins=config_plot[0],
                                        log_scale=config_plot[1])

    except Exception as e:
        pytest.fail(
            f"Your function raised an exception with config {config}: {str(e)}")