# Adversarial attacks on tree-based classifiers

Link to notes:

<https://docs.google.com/document/d/1xqCDqrakSaXkOxijwse3zQ23Rm7-38m-rMJp7iLjS6A/edit?usp=sharing>

Descriptions of datasets:

<https://augustauniversity.box.com/s/ioj5v4ddiwcs9fs451rppzf5vhzohfpo>

## Setup

This repo includes submodules. Clone it, [including its submodules](https://stackoverflow.com/a/4438292).

Required Python environment: 3.8 or 3.9 [^1]

[^1]: This is a hard requirement. Numpy requires >= 3.8; robust XGBoost relies on Python features removed after 3.9.

Install required dependencies:

```
python -m pip install -r requirements.txt
```

### Build and Install Robust XGBoost

By default, attacks are configured to use a modified version of XGBoost classifier, enhanced with adversarial robustness
property. This classifier is not installed with the other package dependencies.

The XGBoost classifier must be built locally [from source](./RobustTrees) (also included as a submodule `RobustTrees`),
Following the [instructions here](./RobustTrees/tree/master/python-package#from-source) to build it from source.

After local build, set up Python environment to use this classifier in experiments:

```
python -m pip install -e "/absolute/local/path/to/RobustTrees/python-package"
```

## Usage

This application is intended for use over command line interface.
There are two execution modes: `experiment` and `validate`.

- `experiment` mode trains a classifier and performs adversarial
  attack according to provided arguments.
- `validate` mode will check a dataset for correctness.

Standard interaction:

```
python3 -m src {experiment|validate} [ARGS]
```

To see available options for experiments, run:

```
python3 -m src experiment --help
```

To see available options for the validator, run:

```
python3 -m src validate --help
```


## Source code organization

| Directory           | Description                               |
|:--------------------|:------------------------------------------|
| `results`           | Results of various experiments            |
| 　`└─ comparisons`   | Comparison of classifiers, on IoT data    |
| 　`└─ decision_tree` | Adversarial attacks on decision trees     |
| 　`└─ xgboost`       | Adversarial attacks on XGBoost classifier |
| `data`              | Preprocessed data sets                    |
| `src`               | Source code                               |
| 　`└─ __init__.py`   | Python package setup                      |
| 　`└─ __main__.py`   | CLI interface                             |
| 　`└─ attack.py`     | Abstract base class for an attack         |
| 　`└─ classifier.py` | Abstract base class for a classifier      |
| 　`└─ experiment.py` | Runs an attack experiment                 |
| 　`└─ hopskip.py`    | HopSkipJump attack implementation         |
| 　`└─ tree.py`       | Decision tree classifier training         |
| 　`└─ utility.py`    | Shared functionality utilities            |
| 　`└─ validator.py`  | Network dataset  validator                |
| 　`└─ xgb.py`        | XGBoost classifier training               |
| 　`└─ zoo.py`        | ZOO attack implementation                 |

