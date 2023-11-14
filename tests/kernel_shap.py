import pytest
import itertools
from iml.methods.kernel_shap import SHAP
import matplotlib

# Test own implementation of Kernel SHAP
# Execute via: python -m pytest iml/tests/kernel_shap.py

# Define ranges for each parameter
# Init-function  parameters
init_values_data = [["", ""]] #TO BE SPECIFIEDs
init_values_debug = [True]

# Fit-function parameters
fit_values_pred_based = [True]
fit_values_loss_based = [True]
fit_values_class_to_be_explained = [0, 1]
fit_values_apply_WLS = [True, False]

# Get-function Parameters
get_values_local = [True, False]
get_values_save_to_path = [None, "save_tests/shap.csv"]

# T-Test-function Parameters
ttest_values_save_to_path = [None, "save_tests/shap.pkl"]

# Avoid pop-up windows for plots
matplotlib.use('Agg')

# Use itertools to generate all combinations of parameters
configurations_init_fit = list(itertools.product(init_values_data,
                                        init_values_debug,
                                        fit_values_pred_based,
                                        fit_values_loss_based,
                                        fit_values_class_to_be_explained,
                                        fit_values_apply_WLS))

configurations_get_plot = list(itertools.product(get_values_local,
                                                 get_values_save_to_path))

# Test Function
@pytest.mark.parametrize('config', configurations_init_fit)
def test_your_function(config):
    try:
        temp_shap = SHAP(data=config[0],
                         debug=config[1])
        temp_shap.fit(pred_based=config[2],
                      loss_based=config[3],
                      class_to_be_explained=config[4],
                      apply_WLS=config[5])

        for config_get_plot in configurations_get_plot:
            #Test  SHAP
            temp_shap.get_SHAP_values(local=config_get_plot[0],
                                      save_to_path=config_get_plot[1])
            temp_shap.plot_bar(loss_based=False)

            #Tesst SAGE
            temp_shap.get_SAGE_values(save_to_path=config_get_plot[1])
            temp_shap.plot_bar(loss_based=True)
            for ttest_save_to_path in ttest_values_save_to_path:
                temp_shap.get_SAGE_t_test(save_to_path=ttest_save_to_path)

    except Exception as e:
        pytest.fail(
            f"Your function raised an exception with config {config}: {str(e)}")