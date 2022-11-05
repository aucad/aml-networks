# Adversarial attacks on tree-based classifiers

Link to notes:

<https://docs.google.com/document/d/1xqCDqrakSaXkOxijwse3zQ23Rm7-38m-rMJp7iLjS6A/edit?usp=sharing>

Descriptions of datasets:

<https://augustauniversity.box.com/s/ioj5v4ddiwcs9fs451rppzf5vhzohfpo>

## Setup

Required Python environment: 3.8 or 3.9 [^1]

[^1]: This is a hard requirement. Numpy requires >= 3.8; robust XGBoost relies on Python features removed after 3.9.

Install required dependencies:

```
python -m pip install -r requirements.txt
```

For Apple chips, install compatible `tensorflow` separately:

```
python -m pip install tensorflow-macos
```

**Optional**: To visualize XGBoost models, install [graphviz](https://graphviz.org/) and make sure it is in path.

### Setup XGBoost

By default, attacks are configured to use a modified version of XGBoost classifier, enhanced with adversarial robustness
property. This classifier is not installed with the other package dependencies.

The XGBoost classifier must be built locally [from source](./RobustTrees) (also included as submodule `RobustTrees`),
following [instructions here](./RobustTrees/tree/master/python-package#from-source).
  

After local build, set up Python environment to use this classifier version:

```
python -m pip install -e "/absolute/local/path/to/RobustTrees/python-package"
```

## Usage

This application is intended for use over command line interface. Standard interaction:

```
python3 -m src [ARGS]
```

To see available options, run:

```
python3 -m src --help
```

## Source code directory organization

| Directory           | Description                                                 |
|:--------------------|:------------------------------------------------------------|
| `results`           | Results of various experiments                              |
| 　`└─ comparisons`   | Comparison of classifiers, on IoT data, obtained using Weka |
| 　`└─ decision_tree` | Adversarial attacks on decision trees                       |
| 　`└─ xgboost`       | Adversarial attacks on XGBoost classifier                   |
| `data`              | Preprocessed data sets, ready for running various attacks   |
| `src`               | Source code                                                 |
| 　`└─ __init__.py`   | Python package setup                                        |
| 　`└─ __main__.py`   | CLI interface                                               |
| 　`└─ attack.py`     | Abstract base class for an attack                           |
| 　`└─ cls.py`        | Abstract base class for a classifier                        |
| 　`└─ experiment.py` | Runs an attack experiment                                   |
| 　`└─ hopskip.py`    | HopSkipJump attack implementation                           |
| 　`└─ plot.py`       | Plots figures                                               |
| 　`└─ tree.py`       | Decision tree classifier training                           |
| 　`└─ utility.py`    | Shared functionality utilities                              |
| 　`└─ validator.py`  | Post-attack record validator                                |
| 　`└─ xgb.py`        | XGBoost classifier training                                 |
| 　`└─ zoo.py`        | ZOO attack implementation                                   |
