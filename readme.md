# Evaluating AML Threats in NIDS

This repository contains an implementation of an _evaluation system_ to run experiments. An experiment measures success rate of adversarial machine learning (AML) evasion attacks in network intrusion detection systems (NIDS).

The system allows to evaluate classifiers trained on network data sets against adversarial black-box evasion attacks. 
Supported classifiers are Keras deep neural network and a tree-based ensemble learner XGBoost. 
Both classifiers can be enhanced with an adversarial defense.

<br/>

**Experiment Overview**

<pre>
┌───────────────┐          ┌───────────┐         ┌───────────────┐          
│  classifier   │ ───────→ │  Evasion  │ ──────→ │   validator   │ ──────→  valid & evasive
│  (+defense)   │ ◁------- │  attack   │         │ [constraints] │          adversarial
└───────────────┘   query  └───────────┘         └───────────────┘          examples                       
 Training data              Testing data 
</pre>

* The input is NIDS data and the validator is configured to know the applicable domain constraints.
* One experiment run repeats 5 times, for different disjoint partitions of the input data, using 5-folds cross-validation.
* Various metrics are recorded: classifier accuracy, evasions before and after validators, time, etc.

<br/>

**Source code organization**

| Directory     | Description                                           |
|:--------------|:------------------------------------------------------|
| `.github`     | Actions workflow files                                |
| `aml`         | Evaluation system implementation source code          |
| `config`      | Experiment configuration files                        |
| `data`        | Preprocessed datasets ready for experiments           |
| `result`      | Referential result for comparison                     |
| `RobustTrees` | (submodule) XGBoost enhanced with adversarial defense |

**Datasets**

- [Aposemat IoT-23][IOT] contains IoT network traffic.
- [UNSW-NB15][UNS] contains traditional network intrusion data.


## Getting Started

The easiest way to run experiments is with [Docker][DOC].
The docker build assumes amd64-compatible host. Otherwise, [build from source](#native-execution).

#### 1. Clone repo and build a container image

```
git clone https://github.com/aucad/aml-networks.git && cd aml-networks
docker build -t aml-networks . 
```

#### 2. Launch the container

```
docker run -v $(pwd)/output:/usr/src/aml-networks/output -it --rm aml-networks /bin/bash
```

## Reproduce Paper Experiments

The runtime estimates are for 8-core 32 GB RAM Linux (Ubuntu 20.04) machine. Actual times vary.

#### 1. Limited model queries 

```
make query
```

(~24h) This experiment uses the full testing set and repeats experiments with different model query limits. 
By default, the max query limits are: 2, 5, default (varies by attack). 

#### 2. Limited sampled input 

```
make sample
```

(~90 min) Run experiments using limited input size by randomly sampling the testing set. 
By default, the sample size is 50 and sampling is repeated 3 times. The result is the average of 3 runs.

#### 3. Plot results 

```
make plots
```

(< 1 min) Plot results of the two previous experiments. The plot data source is `output/` directory. 


## Run Custom Experiments

There are three execution modes:

```
experiment - Performs adversarial attack experiments
plot       - Generate tables from captured experiment results
validate   - Check a dataset for network protocol correctness
```

Custom experiments can be defined by constructing appropriate commands.

```
python3 -m aml {experiment|plot|validate} [ARGS]
```

To see available options for experiments, run:

```
python3 -m aml experiment --help
```

To see available options for plotting results, run:

```
python3 -m aml plot --help
```

To see available options for the validator, run:

```
python3 -m aml validate --help
```

## Native Execution

These steps explain how to run experiments from source natively on host machine.
You should also follow these steps, if you want to prepare a development environment and make code changes.

**Step 0: Environment setup**

- :snake: **Required** Python environment: 3.8 or 3.9

- :warning: **Submodule** This repository has a submodule. Clone it including the submodule:

  ```
  git clone --recurse-submodules https://github.com/aucad/aml-networks.git
  ```

This implementation is not compatible with Apple M1 machines due to underlying dependency (tensorflow-macos); and
although it does not prevent most experiments, some issues may surface periodically.

**Step 1: Build robust XGBoost**

The evaluation uses a modified version of XGBoost classifier, enhanced with adversarial robustness property. 
This classifier is not installed with the other package dependencies and must be built locally from source, i.e. the submodule `RobustTrees`.
By default, you will need gcc compiler with OpenMP support. 
To build robust XGBoost, run:

```
cd RobustTrees
make -j4
```

If the build causes issues, follow [these instructions][RBT] to build it from source.

**Step 2: Install dependencies**

Install required Python dependencies.

```
python3 -m pip install -r requirements-dev.txt
```

Install XGBoost from the local build location:

```
python3 -m pip install -e "/path/to/RobustTrees/python-package"
```

**Step 3: (optional) Check installation**


Check the xgboost runtime, the version number should be 0.72.

```
python3 -m pip show xgboost
```

Run a help command, which should produce a help prompt.

```
python3 -m aml
```

You are ready to run experiments and make code changes. 

[IOT]: https://www.stratosphereips.org/datasets-iot23/
[UNS]: https://research.unsw.edu.au/projects/unsw-nb15-dataset
[DOC]: https://docs.docker.com/engine/install/
[RBT]: https://github.com/chenhongge/RobustTrees/tree/master/python-package#from-source
