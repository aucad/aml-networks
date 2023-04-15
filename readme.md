# Adversarial Machine Learning in NIDS

This repository implements an evaluation infrastructure to measure success rate of adversarial machine learning evasion attacks in network intrusion detection systems (NIDS). 

It involves evaluation of classifiers — trained on network data sets of benign and malicious traffic flows — against adversarial black-box attacks. The currently supported classifiers are: Keras deep neural network, and a tree-based ensemble learner XGBoost. Both classifiers can be enhanced with an adversarial defense during training phase.

This repository provides an implementation to perform various experiments in the specified setting. Instructions for running pre-defined experiments, and extended custom usage, is explained below.

## Getting Started

These steps explain how to build and run experiments from source.

- :snake: **Required** Python environment: 3.8 or 3.9

- :warning: **Submodule** This repository has a submodule. Clone it, including the submodule
  [(instructions)](https://stackoverflow.com/a/4438292).

This implementation is not compatible with Apple M1 chip hosts due to a dependency. Use a machine with x86 architecture.


**Datasets**: we consider two network traffic captures:

1. [Aposemat IoT-23](https://www.stratosphereips.org/datasets-iot23/) is a labeled dataset with malicious and benign IoT network traffic.

2. [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) is a network intrusion dataset that contains nine different attacks.

Preprocessed and sampled data is included in `data/` directory.

### Step 1: Build robust XGBoost

The evaluation uses a modified version of XGBoost classifier, enhanced with adversarial robustness property. This
classifier is not installed with the other package dependencies. The XGBoost classifier must be built
locally [from source](https://github.com/chenhongge/RobustTrees) (also included as a submodule `RobustTrees`). Follow
the
[instructions here](https://github.com/chenhongge/RobustTrees/tree/master/python-package#from-source) to build it from
source.

### Step 2: Install dependencies

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

This application is intended for use over command line interface (CLI). There are three execution modes:

| Mode         | Description                                                                              |
|:-------------|:-----------------------------------------------------------------------------------------|
| `experiment` | Trains a classifier and performs adversarial attack according to specified configuration |
| `plot`       | Generates tables from captured experiment results                                        |
| `validate`   | Check a data set for network protocol correctness                                        |

### Quick start: Predefined Experiments


**Full evaluation**

```
make all
```

This experiment uses full cross-validation holdout set and repeats experiments for different max iterations. Max
iterations can be customized by appending to the command `ITERS`, e.g., `ITERS="5 10 20"`. 
<br/>:warning: &nbsp; This experiment takes 24h on 8-core 32 GB RAM Linux machine.

**Random sample of limited input**

```
make sample
```

Perform experiments on _limited input size_, by randomly sampling records of the holdout set. The sample size can be
customized by appending to the command `SAMPLE=m TIMES=n`, where `m` is the number of records to use and `n` is the
number of times to repeat the sampling. The result is reported as the average of `n` runs.
<br/>:warning: &nbsp; Running this experiment on 8 core/32 GB RAM Linux machine takes ~90 minutes.

**Run a "quick" experiment**

```
make fast
```

This option runs a quick experiment, for a configuration of sampled records and a low iteration count. 
These parameters are fixed: it does not accept arguments like the two experiments above.

**Plot results**

```
make plot
```

Plot results of a previously performed experiment. 

### Configuring Additional Experiments

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
