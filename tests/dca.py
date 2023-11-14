import pytest
import itertools
from iml.methods.dca import DCA
import matplotlib
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier 

# Test Decision Curve Analysis
# Execute via: python -m pytest iml/tests/dca.py

# Define ranges for each parameter
# Init-function  parameters
init_values_data = [["", ""]] #TO BE SPECIFIED
init_values_debug = [True]

# Fit-function parameters
# fit_values_marker = [init_values_data.data.X_train_df.columns[1], init_values_data.data.X_train_df.columns[2], init_values_data.data.X_train_df.columns[3]]
fit_values_random_forest = [True, False]
fit_values_gradient_boosting = [True, False]
fit_values_lightgbm = [False]
fit_values_ascertain_association = [True, False]

# Add-function parameters
add_values_predictor = [linear_model.LogisticRegression()]

# Plot-function parameters
plot_values_predictors = [["TabPFN"], ["TabPFN", "LogReg"]]

# Avoid pop-up windows for plots
matplotlib.use('Agg')

# Use itertools to generate all combinations of parameters
configurations_init_fit = list(itertools.product(init_values_data,
                                        init_values_debug,
                                        fit_values_random_forest,
                                        fit_values_gradient_boosting,
                                        fit_values_lightgbm,
                                        fit_values_ascertain_association,
                                        add_values_predictor,
                                        plot_values_predictors))


@pytest.mark.parametrize('config', configurations_init_fit)

# Test Function
def test_your_function(config):
    try:
        temp_dca = DCA(data=config[0],
                                debug=config[1])

        temp_dca.fit(marker=config[0].X_train_df.columns[1],
                    random_forest= config[2],
                    gradient_boosting= config[3],
                    lightgbm= config[4],
                    ascertain_association=config[5])

        temp_dca.add_predictor(predictor_name="LogReg",
                               predictor=config[6])
        
        temp_dca.plot(predictors=config[7])

    except Exception as e:
        pytest.fail(
            f"Your function raised an exception with config {config}: {str(e)}")
