# Adversarial attacks on tree-based classifiers

Link to notes:

<https://docs.google.com/document/d/1xqCDqrakSaXkOxijwse3zQ23Rm7-38m-rMJp7iLjS6A/edit?usp=sharing>

## Setup

Required Python environment: 3.8-3.9 [^1]

[^1]: this is a hard requirement. numpy requires >= 3.8; 
robust XGBoost relies on Python features removed since 3.10.

Install required dependencies:

```
python -m pip install -q -r requirements.txt
```

For Apple M1/M2 chips, install compatible `tensorflow` separately:

```
python3 -m pip install -q tensorflow-macos
```

To visualize XGBoost models, install [graphviz](https://graphviz.org/) and make sure it is in path.

### Setup XGBoost

By default, attacks are configured to use a modified version of XGBoost classifier,
enhanced with adversarial robustness property.

The XGBoost classifier must be built locally [from source](https://github.com/chenhongge/RobustTrees), 
following [instructions here](https://github.com/chenhongge/RobustTrees/tree/master/python-package#from-source).

After local build, set up Python environment to use this classifier version:

```
python -m pip install -e "/absolute/local/path/to/RobustTrees/python-package"
```

## Run attacks

With default options

```
python src/run_attacks.py
```

Use specific dataset:

```
python src/run_attacks.py ./path/to/input_data.csv
```

## Directory organization

| Directory        | Description                                                 |
|:-----------------|:------------------------------------------------------------|
| `adversarial_dt` | Results of adversarial attacks on decision trees            |
| `adversarial_xg` | Results of adversarial attacks on XGBoost classifier        |
| `comparisons`    | Comparison of classifiers, on IoT data, obtained using Weka |
| `data`           | Preprocessed data sets for running various attacks          |
| `src`            | Source code files                                           |
