# Interpretable Machine Learning for TabPFN

This repo contains all scripts for the standalone TabPFN interpretable ML package.

The project revolves around [TabPFN](https://github.com/automl/TabPFN) with the corresponding [paper](https://arxiv.org/pdf/2207.01848.pdf).

This package is an extensive and easy-to-use toolbox of interpretability methods to be used in conjunction with TabPFN. It aims at leveraging TabPFN, which inherently possesses high predictive power, to explain its predictions.

To account for the peculiarities of TabPFN, many methods are implemented from scratch, as a significant speedup relative to existing open-source implementations could be achieved. For some methods, this package relies on packages by others, in particular: 
- [`shap`](https://github.com/shap/shap) by Scott Lundberg and collaborators.
- [`dcurves`](https://github.com/ddsjoberg/dcurves) by Daniel Sjoberg and collaborators.
- [`MAPIE`](https://github.com/scikit-learn-contrib/MAPIE) by Vianney Taquet, Grégoire Martinon and collaborators.

Moreover, within [tabpfniml/tabpfn_interpret](tabpfniml/tabpfn_interpret), it contains a modified copy of the [TabPFN](https://github.com/automl/TabPFN) code.

This package provides the following methods:

### Feature Effect Methods:
- Individual Conditional Expectation (ICE) curves (local)
- Partial Dependence (PD) plots (global)
- Accumulated Local Effects (ALE) plots (local)
- Kernel SHAP (local & global)
- Sensitivity Analysis (local & global)
### Feature Importance Methods:
- Leave-One-Covariate-Out (LOCO) (global)
- Shapley Additive Global Importances (SAGE) (global)
### Local Explanations:
- Counterfactual Explanations (CE) (local)
### Clinical value assessment & Uncertainty quantification:
- Decision Curve Analysis (DCA) (global)
- Conformal Prediction (local)


# Setup

To test this package, create a new conda environment. To do so, change the environment name `package_test` to a name of your choice and run the following:
```shell
conda create --name package_test python=3.8 pip
conda activate package_test
```
Now, install the package tabpfn_iml from the local build file using pip into the conda environment:

```shell
pip install dist/tabpfniml-0.0.1.tar.gz
```

If it doesn't work (due to Python PATH conflicts), use:

```shell
/Users/<username>/miniconda3/envs/package_test/bin/pip install /Users/<username>/<path_to_repo>/tabpfn_iml/dist/tabpfniml-0.0.1.tar.gz
```
After successful installation, you can import methods from the package like this:
```python
from tabpfniml import ICE_PD
```

# Datasets

A new dataset instance that inherits from the `dataset_iml` class must be instantiated in order to apply the IML methods. Essentially you only need to define the `__init__()` function and please refer to `demo/consulting_datasets/datasets.py` for the structure and necessary variables. The datasets needs to be a *.csv* file and the first columns needs to be the target and the other colums the features. Categorical features need to be already converted to integer or float values, e.g., by label encoding.

```python
from tabpfniml import dataset_iml

class myOwnData(dataset_iml):
  def __init__(self):
    ...

```

