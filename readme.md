# Adversarial Machine Learning in NIDS

This repository implements an evaluation infrastructure to measure success rate of adversarial machine learning evasion
attacks in network intrusion detection systems (NIDS).

It involves evaluation of classifiers — trained on network data sets of benign and malicious traffic flows — against
adversarial black-box attacks. The currently supported classifiers are Keras deep neural network, and a tree-based
ensemble learner XGBoost. Both classifiers can be enhanced with an adversarial defense.

This repository provides an implementation to perform various experiments in the specified setting. Instructions for
running pre-defined experiments, and extended custom usage, is explained below.

**Repository Organization**

- `config`     — Experiment configuration files              
- `data`       — Preprocessed data sets ready for experiments
- `ref_result` — Referential result for comparison          
- `src`        — Implementation source code                  

**Datasets**: we consider two network traffic captures:

1. [Aposemat IoT-23](https://www.stratosphereips.org/datasets-iot23/) is a labeled dataset with malicious and benign IoT
   network traffic.

2. [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) is a network intrusion dataset that contains
   nine different attacks.

## Getting Started

The easiest way to run various experiments is using a [Docker](https://docs.docker.com/engine/install/) container.
The container has all environment dependencies pre-configured.

**Build a container** - make sure to include the dot

```
docker build -t aml-networks .
```

**Launch the container** - add output directory to persist experiment results

```
# create directory for experiment results
mkdir output

# Launch the container
docker run -v $(pwd)/output:/aml-networks/output -it --rm aml-networks /bin/bash
```

Once the container is running, you are ready to run experiments, as explained in the next section.

## Usage: Running Predefined Experiments

The runtime estimates are for 8-core/32 GB RAM Linux machine.

**Full evaluation** ~ 24h

```
make all
```

This experiment uses full cross-validation holdout set and repeats experiments for different max iterations. Max
iterations can be customized by appending to the command e.g., `ITERS="5 10 20"`.

**Random sample of limited input** ~ 90 min

```
make sample
```

Perform experiments on _limited input size_, by randomly sampling records of the holdout set. The sample size can be
customized by appending to the command `SAMPLE=m TIMES=n`, where $m$ is the number of records to use and $n$ is the
number of times to repeat the sampling. The result is reported as average of $n$ runs.

**Plot results** < 1 min

```
make plot
```

Plot results of a previously performed experiment. 
By default, the plot data source is `output` directory. 
To plot some other directory append `RES_DIR` to the command,
e.g. `RES_DIR=ref_result/all`.

## Usage: Running Custom Experiments

There are three execution modes:

| Mode         | Description                                                                              |
|:-------------|:-----------------------------------------------------------------------------------------|
| `experiment` | Trains a classifier and performs adversarial attack according to specified configuration |
| `plot`       | Generates tables from captured experiment results                                        |
| `validate`   | Check a data set for network protocol correctness                                        |

Custom experiments can be defined by constructing appropriate commands.

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

---

## Development Setup and Running Natively on Host

These steps explain how to run experiments from source.
You should also follow these steps to prepare a development environment.

- :snake: **Required** Python environment: 3.8 or 3.9

- :warning: **Submodule** This repository has a submodule. Clone it, including the submodule
  [(instructions)](https://stackoverflow.com/a/4438292).

This implementation is not compatible with Apple M1 chip hosts due to a dependency. Use a machine with x86 architecture.

### Step 1: Build robust XGBoost

The evaluation uses a modified version of XGBoost classifier, enhanced with adversarial robustness property. This
classifier is not installed with the other package dependencies. The XGBoost classifier must be built
locally [from source](https://github.com/chenhongge/RobustTrees) (also included as a submodule `RobustTrees`). Follow
the
[instructions here](https://github.com/chenhongge/RobustTrees/tree/master/python-package#from-source) to build it from
source.

### Step 2: Install dependencies

Install required Python dependencies.

```
python3 -m pip install -r requirements-dev.txt
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