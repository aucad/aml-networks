# Adversarial attacks in networks

This repository implements an evaluation pipeline to measure success rate of adversarial machine learning
evasion attacks in network settings.

It involves evaluation of two network data sets and two adversarial black-box attacks, on a
tree-based ensemble learner XGBoost. The classifier is enhanced with an adversarial robustness property. The classifier
must be [built from source](#build-robust-xgboost) before performing any experiments. 

This repository then provides an implementation to perform various experiments in the specified adversarial setting.
Instructions for running pre-defined experiments, and extended custom usage, is explained in section [usage](#usage).

## Datasets

We consider two data sources:

1. [Aposemat IoT-23](https://www.stratosphereips.org/datasets-iot23/) is a labeled dataset with malicious and benign IoT network traffic.

2. [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) is a network intrusion dataset that contains nine different attacks.

Preprocessed, sampled data is included in `data/` directory. The input data is expected to be in csv format.

## Setup

These steps explain how to run this software from source.

Required Python environment: 3.8 or 3.9 [^1]

[^1]: This is a hard requirement. Numpy requires >= 3.8; robust XGBoost relies on Python features removed after 3.9.

This repository includes a submodule. 
Clone it including the [submodule](https://stackoverflow.com/a/4438292).

### Build robust XGBoost

By default, attacks are configured to use a modified version of XGBoost classifier, enhanced with adversarial robustness
property. This classifier is not installed with the other package dependencies.
The XGBoost classifier must be built locally [from source](https://github.com/chenhongge/RobustTrees) (also included as a submodule `RobustTrees`).
Follow the [instructions here](https://github.com/chenhongge/RobustTrees/tree/master/python-package#from-source) to build it from source.

### Install dependencies

Install required Python dependencies:

```
python3 -m pip install -r requirements.txt
```

Install XGBoost from the local build location:

```
python3 -m pip install -e "/path/to/RobustTrees/python-package"
```

After setup, check the runtime:

```
python3 -m pip show xgboost
```

The version number should be 0.72.


## Usage

This application is intended for use over command line interface.
There are 3 execution modes: `experiment`, `plot` and `validate`.

- `experiment` mode trains a classifier and performs adversarial
  attack according to provided arguments.
- `plot` mode generates tables from captured experiment results.
- `validate` mode will check a dataset records for correctness.

**Quick start**

Run experiments

```
make all
```

The default experiment uses full cross-validation holdout set, and
repeats experiments for three options of max iterations.

There are two alternative pre-defined experiments:

- `make sample` runs the experiment on limited input size of 50 records.
- `make proto` subset of all; runs experiment for default iterations only.

Custom experiments can be defined using the CLI commands.

Plot results

```
make plot
```

Validate datasets

```
make valid
```

**CLI commands**

```
python3 -m src {experiment|plot|validate} [ARGS]
```

To see available options for experiments, run:

```
python3 -m src experiment --help
```

To see available options for plotting results, run:

```
python3 -m src plot --help
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
| 　`└─ machine.py`    | Captures current machine details          |
| 　`└─ plotter.py`    | Experiment results plotting               |
| 　`└─ tree.py`       | Decision tree classifier training         |
| 　`└─ utility.py`    | Shared functionality utilities            |
| 　`└─ validator.py`  | Network dataset  validator                |
| 　`└─ xgb.py`        | XGBoost classifier training               |
| 　`└─ zoo.py`        | ZOO attack implementation                 |
