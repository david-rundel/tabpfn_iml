import pytest
import itertools
from iml.methods.ice_pd import ICE_PD
import matplotlib

# Test ICE and PD
# Execute via: python -m pytest iml/tests/ice_pd.py

# Define ranges for each parameter
# Init-function  parameters
init_values_data = [["", ""]] #TO BE SPECIFIED
init_values_debug = [True]

# Fit-function parameters
fit_values_features=[None, [0]]
fit_values_max_levels_per_feature = [5, 10]

# Get-PD-function parameters
get_values_feature = [0]
get_values_save_to_path = [None, "save_tests/pd.csv"]

# Plot-function parameters
plot_values_features = [None, [0]]
plot_values_plot_classification_threshold = [False, True]
plot_values_center = [False, True]
plot_values_save_to_dir = [None, "save_tests"]

# Avoid pop-up windows for plots
matplotlib.use('Agg')

configurations_is_init_fit = list(itertools.product(init_values_data,
                                                    init_values_debug,
                                                    fit_values_features,
                                                    fit_values_max_levels_per_feature))   
configurations_is_get = list(itertools.product(get_values_feature,
                                               get_values_save_to_path))
configurations_is_plot = list(itertools.product(plot_values_features,
                                               plot_values_plot_classification_threshold,
                                               plot_values_center,
                                               plot_values_save_to_dir))

# Test Function
@pytest.mark.parametrize('config', configurations_is_init_fit)
def test_your_function(config):
    try:
        temp_ice = ICE_PD(data=config[0],
                         debug=config[1])
        temp_ice.fit(features=config[2],
                     max_levels_per_feature=config[3])

        for config_get in configurations_is_get:
            temp_ice.get_PD(feature=config_get[0],
                            save_to_path=config_get[1])
        
        for config_plot in configurations_is_plot:
            temp_ice.plot(features=config_plot[0],
                            plot_classification_threshold=config_plot[1],
                            center=config_plot[2],
                            save_to_dir=config_plot[3])
            
    except Exception as e:
        pytest.fail(
            f"Your function raised an exception with config {config}: {str(e)}")