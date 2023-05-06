# Evaluating AML Threats in NIDS

This repository implements an evaluation infrastructure to measure success rate of adversarial machine learning evasion
attacks in network intrusion detection systems (NIDS).

It enables evaluating classifiers, trained on network data sets of benign and malicious traffic flows, against 
adversarial black-box evasion attacks. Supported classifiers are Keras deep neural network and a tree-based ensemble 
learner XGBoost. Both classifiers can be enhanced with an adversarial defense.

This repository provides an implementation to perform various experiments in this specified setting. 

**Repository Organization**

- `config`     — Experiment configuration files              
- `data`       — Preprocessed datasets ready for experiments
- `result`     — Referential result for comparison          
- `src`        — Implementation source code                  

**Datasets**

- [Aposemat IoT-23](https://www.stratosphereips.org/datasets-iot23/) contains IoT network traffic.
- [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) contains traditional network intrusion data.

## Getting Started

The easiest way to run these experiments is using [Docker](https://docs.docker.com/engine/install/).

1. Create an output directory to persist experiment results

    ```
    mkdir output
    ```
   
2. Build a container

    ```
    docker build -t aml-networks .
    ```

3. Launch the container

    ```
    docker run -v $(pwd)/output:/aml-networks/output -it --rm aml-networks /bin/bash
    ```

## Run Paper Experiments

The runtime estimates are for 8-core 32 GB RAM Linux machine, actual time may vary.

:one: **Limited model queries** ~24h

```
make query
```

This experiment uses full cross-validation holdout set and repeats experiments using different model query limits. 
By default it runs attacks with limits 2, 5, and default iterations. Query limits can be customized by appending 
to the command a limit argument, for example `LIMIT="5 10 20"`.

:two: **Random sampling of limited input** ~90 min

```
make sample
```

Perform experiments on limited input size, by randomly sampling records of the holdout set. The sample size can be
customized by appending to the command `SAMPLE=m TIMES=n`, where $m$ is the number of records to use and $n$ is the
number of times to repeat the sampling. The result is reported as average of $n$ runs. Model query limit is unset,
meaning attack's default limit is used.


:eight_pointed_black_star: **Plot results** < 1 min

```
make plot
```

Plot results of a previously performed experiment. The plot data source defaults to `output` directory. 
To plot some other directory append `RESDIR` to the command, e.g. `make plot RESDIR=result/all`.

## Run Custom Experiments

There are three execution modes:

```
experiment - Performs adversarial attack experiments
plot       - Generate tables from captured experiment results
validate   - Check a dataset for network protocol correctness
```

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

## About this Repository

```
github.com/AlDanial/cloc v 1.96  T=0.19 s (420.7 files/s, 275235.9 lines/s)
-------------------------------------------------------------------------------
Language                     files          blank        comment           code
-------------------------------------------------------------------------------
CSV                              3              0              0          30003
JSON                            56              0              0          20414
Python                          13            320            289           1554
Markdown                         2             73              0            173
make                             1             18              0             52
YAML                             3              0             29             31
Dockerfile                       1              4              0             18
Text                             2              1              0             13
-------------------------------------------------------------------------------
SUM:                            81            416            318          52258
-------------------------------------------------------------------------------
```

## Native Execution

These steps explain how to run experiments from source natively on host machine.
You should also follow these steps, if you want to prepare a development environment and make code changes.

- :snake: **Required** Python environment: 3.8 or 3.9

- :warning: **Submodule** This repository has a submodule. Clone it, including the submodule
  [(instructions)](https://stackoverflow.com/a/4438292).

This implementation is not compatible with Apple chip hosts due to a dependency (issue with TensorFlow). 
Use a machine with x86 architecture.

**Step 1: Build robust XGBoost**

The evaluation uses a modified version of XGBoost classifier, enhanced with adversarial robustness property. This
classifier is not installed with the other package dependencies. The XGBoost classifier must be built
locally [from source](https://github.com/chenhongge/RobustTrees) (also included as a submodule `RobustTrees`). 
Follow the [instructions here](https://github.com/chenhongge/RobustTrees/tree/master/python-package#from-source) 
to build it from source.

**Step 2: Install dependencies**

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