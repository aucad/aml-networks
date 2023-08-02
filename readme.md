# Evaluating AML Threats in NIDS

Implementation of an evaluation system, to measure success rate of adversarial machine learning (AML) evasion
attacks, in network intrusion detection systems (NIDS).

The setup allows to evaluate classifiers, trained on network data sets, against adversarial black-box evasion attacks. 
Supported classifiers are Keras deep neural network and a tree-based ensemble learner XGBoost. 
Both classifiers can be enhanced with an adversarial defense.

**Source code organization**

| Directory     | Description                                           |
|:--------------|:------------------------------------------------------|
| `.github`     | Actions workflow files                                |
| `aml`         | Implementation source code                            |
| `config`      | Experiment configuration files                        |
| `data`        | Preprocessed datasets ready for experiments           |
| `result`      | Referential result for comparison                     |
| `RobustTrees` | (submodule) XGBoost enhanced with adversarial defense |

**Datasets**

- [Aposemat IoT-23](https://www.stratosphereips.org/datasets-iot23/) contains IoT network traffic.
- [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) contains traditional network intrusion data.

## Getting Started

The easiest way to run these experiments is using [Docker](https://docs.docker.com/engine/install/).

1. **Pull the pre-built container image**

    ```
    docker pull ghcr.io/iotcad/aml-networks:latest
    ```

2. **Launch the container**

    ```
    docker run -v $(pwd)/output:/usr/src/aml-networks/output -it --rm ghcr.io/iotcad/aml-networks:latest /bin/bash
    ```

<details>
<summary>Alternatively, build a container locally.</summary>

<br/>

<p align="center">
<i>** Docker environment assumes an amd64 host machine. 
For other hosts, <a href="#native-execution">build from source</a>. **</i>
</p>

1. Clone repository

   ```
   git clone https://github.com/iotcad/aml-networks.git
   ```

2. Build the container

   ```
   cd aml-networks && docker build -t aml-networks . & cd ..
   ```

3. Run the container

   ```
   docker run -v $(pwd)/output:/usr/src/aml-networks/output -it --rm aml-networks /bin/bash
   ```

</details>

## Reproduce Paper Experiments

The runtime estimates are for 8-core 32 GB RAM Linux (Ubuntu 20.04) machine. Actual times vary.

#### 1. Limited model queries 

```
make query
```

(~24h) This experiment uses full cross-validation holdout set and repeats experiments using different model query limits. 
By default, it runs attacks with limits 2, 5, and default iterations. 

#### 2. Limited sampled input 

```
make sample
```

(~90 min) Perform experiments on limited input size, by randomly sampling records of the holdout set. 
By default, the sample size `n`=50 and sampling is repeated 3 times. The result is reported as average of `n` runs. 
Model query limit is the attack's default query limit.

#### 3. Plot results 

```
make plots
```

(< 1 min) Plot results of the two previous experiments. The plot data source is the `output` directory. 


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

This implementation is not compatible with Apple M1 machines due to underlying dependency (tensorflow-macos); and
although it does not prevent most experiments, some issues may surface periodically.

- :snake: **Required** Python environment: 3.8 or 3.9

- :warning: **Submodule** This repository has a submodule. Clone it [including the submodule](https://stackoverflow.com/a/4438292) typically:

  ```
  git clone --recurse-submodules https://github.com/iotcad/aml-networks.git
  ```

**Step 1: Build robust XGBoost**

The evaluation uses a modified version of XGBoost classifier, enhanced with adversarial robustness property. This
classifier is not installed with the other package dependencies and must be built
locally from source (the submodule `RobustTrees`).

By default, you will need gcc compiler with OpenMP support. Assuming a suitable environment, run:

```
cd RobustTrees
make -j4
```

If the build causes issues, follow [these instructions](https://github.com/chenhongge/RobustTrees/tree/master/python-package#from-source)
to build it from source, and explore the various build options for different environments.


**Step 2: Install dependencies**

Install required Python dependencies.

```
python3 -m pip install -r requirements-dev.txt
```

Install XGBoost from the local build location:

```
python3 -m pip install -e "/path/to/RobustTrees/python-package"
```

After setup, check the xgboost runtime:

```
python3 -m pip show xgboost
```

The version number should be 0.72.

Next, run:

```
python3 -m aml
```

This should produce a help prompt with instructions.
You are now ready to run experiments and make code changes. 
