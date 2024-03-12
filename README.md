# Interpretable Machine Learning for TabPFN

This repo contains all scripts for the standalone TabPFN interpretable ML package.

The project revolves around [TabPFN](https://github.com/automl/TabPFN) with the corresponding [paper](https://arxiv.org/pdf/2207.01848.pdf).

This package is an extensive and easy-to-use toolbox of interpretability methods to be used in conjunction with TabPFN. It aims at leveraging TabPFN, which inherently possesses high predictive power, to explain its predictions.

To account for the peculiarities of TabPFN, many methods are implemented from scratch, as a significant speedup relative to existing open-source implementations could be achieved. For some methods, this package relies on packages by others, in particular: 
- [`shap`](https://github.com/shap/shap) by Scott Lundberg and collaborators.
- [`dcurves`](https://github.com/ddsjoberg/dcurves) by Daniel Sjoberg and collaborators.
- [`MAPIE`](https://github.com/scikit-learn-contrib/MAPIE) by Vianney Taquet, Grégoire Martinon and collaborators.

Moreover, within [tabpfniml/tabpfn_interpret](tabpfniml/tabpfn_interpret), it contains a modified copy of the [TabPFN](https://github.com/automl/TabPFN) code.

**A demo notebook, containing examples of all methods available in this package, is located at** [demo/tabpfniml_demo_notebook.ipynb](demo/tabpfniml_demo_notebook.ipynb).

This package provides the following methods:

### Feature Effect Methods
| Method Name                                   | Filepath to Implementation            | Scope           |
|-----------------------------------------------|---------------------------------------|-----------------|
| Individual Conditional Expectation (ICE) curves | `tabpfniml/methods/ice_pd.py`         | Local           |
| Partial Dependence (PD) plots                | `tabpfniml/methods/ice_pd.py`          | Global          |
| Accumulated Local Effects (ALE) plots        | `tabpfniml/methods/ale.py`             | Local           |
| Kernel SHAP                                  | `tabpfniml/methods/kernel_shap.py` & `tabpfniml/methods/kernel_shap_package.py`     | Local & Global  |
| Sensitivity Analysis                         | `tabpfniml/methods/sensitivity.py`     | Local & Global  |


### Feature Importance Methods
| Method Name                             | Filepath to Implementation          | Scope   |
|-----------------------------------------|--------------------------------------|---------|
| Leave-One-Covariate-Out (LOCO)          | `tabpfniml/methods/loco.py`          | Global  |
| Shapley Additive Global Importances (SAGE) | `tabpfniml/methods/kernel_shap.py`    | Global  |



### Data Valuation Methods
| Method Name                            | Filepath to Implementation            | Scope   |
|----------------------------------------|----------------------------------------|---------|
| Leave-One-Out (LOO)                    | `tabpfniml/methods/loco.py`            | Global  |
| Sensitivity Analysis                   | `tabpfniml/methods/sensitivity.py`     | Global  |
| Data Shapley                           | `tabpfniml/methods/data_shapley.py`    | Global  |


### Local Explanations
| Method Name                           | Filepath to Implementation          | Scope  |
|---------------------------------------|--------------------------------------|--------|
| Counterfactual Explanations (CE)      | `tabpfniml/methods/counterfactuals.py` | Local  |



### Clinical value assessment & Uncertainty quantification
| Method Name                           | Filepath to Implementation          | Scope   |
|---------------------------------------|--------------------------------------|---------|
| Decision Curve Analysis (DCA)         | `tabpfniml/methods/dca.py`           | Global  |
| Conformal Prediction                  | `tabpfniml/methods/conformal_pred.py` | Local   |



# Setup

To test this package, create a new conda environment. To do so, change the environment name `package_test` to a name of your choice and run the following:
```shell
conda create --name package_test python=3.8
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

The IML methods can be applied to any dataset from the OpenML Benchmarking Suites using the `OpenMLData` class. Alternatively, the convenience class `ArrayData` can be used to construct a dataset instance from any array that can be transformed into a pandas DataFrame. Moreover, a new dataset instance that inherits from the `dataset_iml` class can be instantiated. Essentially, you only need to define the `__init__()` function. Please refer to [tabpfniml/datasets/datasets.py](tabpfniml/datasets/datasets.py) for the structure and necessary variables. The datasets needs to be a *.csv* file and the first columns needs to be the target and the other colums the features. Categorical features should have already been converted to integer or float values, e.g., by label encoding.

```python
from tabpfniml import dataset_iml

class myOwnData(dataset_iml):
  def __init__(self):
    ...

```

# Experiments

All experiments conducted in the paper can be found in this repository, enabling the straightforward replication.

| Experiment                 | Paper Section    | Directory                            |
|----------------------------|------------------|--------------------------------------|
| ICE & PD                   | 4.1              | `experiments/ice_pd`                 |
| Kernel SHAP                | 4.2              | `experiments/kernel_shap`            |
| Context Optimization       | 4.3              | `experiments/context_optimization`   |
