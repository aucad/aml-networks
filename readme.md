# Adversarial Machine Learning in Network Intrusion Detection Systems

This repository implements an **evaluation pipeline** to measure success rate of adversarial machine learning evasion
attacks in network intrusion detection systems (NIDS). It involves evaluation of select classifiers — trained on network
data sets of benign and malicious traffic flows — against adversarial black-box attacks, and with/out defenses. The
currently supported classifiers are: Keras deep neural network, and a tree-based ensemble learner XGBoost. Both
classifiers can be enhanced with an adversarial robustness during the training phase.

This repository provides an implementation to perform various experiments in the specified setting. Instructions for
running pre-defined experiments, and extended custom usage, is explained in ["Usage" section](#usage) below.

**Datasets**: we consider two network traffic captures:

1. [Aposemat IoT-23](https://www.stratosphereips.org/datasets-iot23/) is a labeled dataset with malicious and benign IoT
   network traffic.

2. [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) is a network intrusion dataset that contains
   nine different attacks.

Preprocessed and sampled data is included in `data/` directory.

## Setup

These steps explain how to run this software from source.

- :snake: **Required** Python environment: 3.8 or 3.9

- :warning: **Submodule** This repository has a submodule. Clone it, including the submodule
  [(instructions)](https://stackoverflow.com/a/4438292).

### Build robust XGBoost

The evaluation uses a modified version of XGBoost classifier, enhanced with adversarial robustness property. This
classifier is not installed with the other package dependencies. The XGBoost classifier must be built
locally [from source](https://github.com/chenhongge/RobustTrees) (also included as a submodule `RobustTrees`). Follow
the
[instructions here](https://github.com/chenhongge/RobustTrees/tree/master/python-package#from-source) to build it from
source.

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

This application is intended for use over command line interface (CLI). There are 3 execution modes: `experiment`
, `plot` and `validate`.

| Mode         | Description                                                                         |
|:-------------|:------------------------------------------------------------------------------------|
| `experiment` | Trains a classifier and performs adversarial attack according to provided arguments |
| `plot`       | Generates tables from captured experiment results                                   |
| `validate`   | Check a data set for network protocol correctness                                   |

### Quick start: Predefined experiments

* `make all`

  Experiment uses full cross-validation holdout set and repeats experiments for different max iterations. Max iterations
  can be customized by specifying: `ITERS="n₀ … nₖ"`. For example: `make all ITERS="5 20 0"`. Value `0` is attack
  default max iterations (varies by attack).


* `make sample`

  Perform experiments on limited input size by randomly sampling records of the holdout set. The sample size can be
  customized by appending to the command `SAMPLE=m TIMES=n`, where `m` is the number of records to use and `n` is the
  number of times to repeat the sampling. The result is reported as the average of `n` runs.

* `make fast`

  Subset of "make all". This option runs experiment for default max iterations only using full holdout set. Parameters
  for this attack are fixed -- it does not accept custom arguments like the two experiments above.

* `make plot`

  plot results of an experiment. This command will generate text-based table plots.

### Additional Custom Experiments

Other custom experiments can be defined by constructing appropriate CLI commands.

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
