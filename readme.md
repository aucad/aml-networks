# Evaluating AML Threats in NIDS

This repository implements an evaluation infrastructure to measure success rate of adversarial machine learning evasion
attacks in network intrusion detection systems (NIDS).

It enables evaluating classifiers, trained on network data sets, against adversarial black-box evasion attacks. 
Supported classifiers are Keras deep neural network and a tree-based ensemble learner XGBoost. Both classifiers can be 
enhanced with an adversarial defense.

This repository provides an implementation to perform various experiments in this specified setting. 

**Source code organization**

| Directory | Description                                 |
|:----------|:--------------------------------------------|
| `aml`     | Implementation source code                  |
| `config`  | Experiment configuration files              |
| `data`    | Preprocessed datasets ready for experiments |
| `result`  | Referential result for comparison           |

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
    docker run -v $(pwd)/output:/usr/src/aml-networks/output \
               -it --rm ghcr.io/iotcad/aml-networks:latest /bin/bash
    ```

**Alternatively,** build a container locally.

<details>
<summary>Container build instructions</summary>

<br/>

_For non-amd64 machines, recommend [building from source](https://github.com/iotcad/aml-networks/blob/main/.github/CONTRIBUTING.md)._ 

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

The runtime estimates are for 8-core 32 GB RAM Linux machine. Actual time may vary.

#### 1️⃣ Limited model queries ~24h

```
make query
```

This experiment uses full cross-validation holdout set and repeats experiments using different model query limits. 
By default, it runs attacks with limits 2, 5, and default iterations. 

#### 2️⃣ Random sampling of limited input ~90 min

```
make sample
```

Perform experiments on limited input size, by randomly sampling records of the holdout set. 
By default, the sample size $n$=50 and sampling is repeated 3 times. The result is reported as average of $n$ runs. 
Model query limit is the attack's default.

#### 3️⃣️ Plot results < 1 min

```
make plots
```

Plot results of the two previous experiments. The plot data source is the `output` directory. 


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

For native execution see [these instructions](https://github.com/iotcad/aml-networks/blob/main/.github/CONTRIBUTING.md).
