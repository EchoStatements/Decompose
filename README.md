# Decompose: A library for computing bias-variance-diversity and bias-variance decompositions

Welcome to Decompose, a library for estimating components of bias-variance and bias-variance-diversity decompositions, including notebooks for replicating experiments in "_A Unified Theory of Diversity in Ensemble Learning_" (https://arxiv.org/abs/2301.03962 )

The library computes decompositions of the loss of ensembles for the following loss functions:
- Squared Loss
- Cross Entropy
- 0-1 Loss
- Poisson Loss
- Logistic Margin Loss
- Exponential Margin Loss

The library also contains a tool for constructing experiments to examine the effects of varying different model parameters
on the components of these decompositions. 

## Installation
These installation instructions and `requirements.txt` have been tested as being correct for Python 3.10.4

### Clone Repository
In the directory where you want to store the repository:
```bash
git clone git@github.com:EchoStatements/Decompose.git 
```

### Install Requirements

After activating virtual environment if necessary, in the top directory of this repository:

**Using Pip**

If using pip as your package manager:

```bash
pip install -r requirements.txt
```

**Using Conda**

Even when using a conda environment, we suggest using the `pip` command above to install the requirements. However, the you can use the following command to install using the conda package manager:

```bash
conda install --file requirements.txt -c conda-forge
```

Correct installation can be verified by opening python in terminal and verifying that `import decompose` runs without error. 

### Install Decompose Library

In the top directory of this repository:

The following command is suitable for use in both conda environments and virtualenv environments:
```bash
pip install -e .
```

Alternatively the following can be used in conda environments

```bash
python setup.py develop
```

## Minimal Working Examples

We present some simple examples of usage of the library. A runnable version this code
can be found in `example_notebooks/Examples_0_Minimal_Examples.ipynb`

### Example 1: Bias-Variance-Diversity Decomposition
In the first example, we show how bias, variance and diversity can be computed for a single model configuration

```python
from sklearn.ensemble import RandomForestRegressor
from decompose import BVDExperiment, plot_bvd
from decompose.data_utils import load_standard_dataset

# Set up model and data
train_data, train_labels, test_data, test_labels = load_standard_dataset("california", 0.5, normalize_data=True)
rf_regressor = RandomForestRegressor(n_estimators=5, max_depth=10)

# Create experiment object
experiment = BVDExperiment(rf_regressor, loss="squared")
# Run experiment on data
results = experiment.run_experiment(train_data, train_labels, test_data, test_labels)
# Print Summary
results.print_summary()

```

This gives the output
```python

Average bias:
[[0.28461199]]

Average variance:
[[0.22549189]]

Diversity:
[[0.17048255]]

Ensemble expected risk:
[[0.33962133]]
```

Note that we did not set random seed here, so different runs will give slightly different results

### Example 2: Effects of Varying Hyperparameters on Bias, Variance and Diversity

In the second example, we show how our framework can be used to show the effect of 
varying a parameter on the bias, variance and diversity.

In this experiment, the maximum depth of the decision trees in a random forest is varied from one to ten, and the bias,
variance and diversity are measured for each model configuration.

```python
# Create experiment object and define parameter to vary
experiment = BVDExperiment(rf_regressor, loss="squared", parameter_name="max_depth",
                           parameter_values=range(1, 11, 1))
# Run experiment on data
experiment.run_experiment(train_data, train_labels, test_data, test_labels)
# Plot Results
plot_bvd(experiment.results_object)
plt.show()
```

![image](experiment_notebooks/images/minimal_example.png)

## Example 3: Using Decomposition Objects

The `BVDExperiment` class is designed to make it is to design and run experiments. 
However, bias variance and diversity can also be computed on experiments from other sources.
Here, we create a synthetic set of labels and predictions, and show how our library can compute  the bias, variance and diversity from this data.

```python
n_trials = 100
ensemble_size = 10
n_examples = 500
n_classes = 3

import numpy as np

# Labels are integers corresponding to the correct class
labels = np.random.choice(n_classes, size=(n_examples,))
# Predictions are in the form of one hot vectors
preds = np.random.uniform(0, 1, (n_trials, ensemble_size, n_examples, n_classes))
preds = preds / preds.sum(axis=3, keepdims=True)

from decompose import CrossEntropy

decomposition = CrossEntropy(preds, labels)

print(f"expected risk: {decomposition.expected_ensemble_loss.mean()}")
print(f"average bias: {decomposition.average_bias.mean()}")
print(f"average variance: {decomposition.average_variance.mean()}")
print(f"diversity: {decomposition.diversity.mean()}")

```
This gives the output:
```
expected risk: 1.1322538359110454
average bias: 1.1035425492110886
average variance: 0.2357285469267884
diversity: 0.20701726022683184
```

A more complete set of examples, demonstrating more features of the library, can be found in the `example_notebooks` directory.

