# Adversarial attacks on tree-based classifiers

Link to notes:

<https://docs.google.com/document/d/1xqCDqrakSaXkOxijwse3zQ23Rm7-38m-rMJp7iLjS6A/edit?usp=sharing>

Descriptions of datasets:

<https://augustauniversity.box.com/s/ioj5v4ddiwcs9fs451rppzf5vhzohfpo>

## Setup

Required Python environment: 3.8-3.9 [^1]

[^1]: this is a hard requirement. numpy requires >= 3.8; robust XGBoost relies on Python features removed since 3.10.

Install required dependencies:

```
python -m pip install -r requirements.txt
```

For Apple chips, install compatible `tensorflow` separately:

```
python -m pip install tensorflow-macos
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

## Directory organization

| Directory            | Description                                                 |
|:---------------------|:------------------------------------------------------------|
| `results`            | Results of various experiments                              |
| 　`└─ comparisons`    | Comparison of classifiers, on IoT data, obtained using Weka |
| 　`└─ decision_tree`  | Adversarial attacks on decision trees                       |
| 　`└─ xgboost`        | Adversarial attacks on XGBoost classifier                   |
| `data`               | Preprocessed data sets, ready for running various attacks   |
| `src`                | Source code                                                 |
| 　`└─ examples/`      | ART examples                                                |
| 　`└─ attack_hop.py`  | Perform HopSkipJump attack                                  |
| 　`└─ attack_zoo.py`  | Perform ZOO attack                                          |
| 　`└─ run_attacks.py` | Run all attacks (robust/non-robust) on some dataset         |
| 　`└─ train_dt.py`    | Train decision tree classifier                              |
| 　`└─ train_xg.py`    | Train XGBoost classifier                                    |
| 　`└─ uitlity.py`     | Common helpers                                              |

## Notes on ART Attacks

See [ART Evasion attacks Wiki](https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/ART-Attacks#1-evasion-attacks)

| Attack                     | Box | Mask | Compatible | Issue Explanation                                                                              |
|:---------------------------|:---:|:----:|:----------:|:-----------------------------------------------------------------------------------------------|
| Adversarial Patch          | ▫️  |  ✔️  |     ➖      | Requires neural network classifier                                                             |
| Auto Attack                | ▪️  |  ➖   |     ➖      | Classifier must support loss type `cross_entropy`                                              |
| Boundary Attack            | ▪️  |  ➖   |     ➖      | This attack has not yet been tested for binary classification with a single output classifier  |
| Carlini and Wagner L0      | ▫️  |  ✔️  |     ➖      | Requires ClassGradientsMixin based classifier                                                  |
| DPatch                     | ▫️  |  ✔️  |     ➖      | Requires classifier derived from `LossGradientsMixin`                                          |
| Frame Saliency Attack      | ▫️  |  ✔️  |     ➖      | Requires neural network classifier                                                             | 
| GeoDA                      | ▪️  |  ➖   |     ➖      | `XGBoostClassifier` object has no attribute `'channels_first'` (same error for decision trees) |
| HopSkipJump                | ▪️  |  ✔️  |     ✔️     |                                                                                                |
| Pixel Attack               | ▪️  |  ➖   |     ➖      | Requires neural network classifier                                                             |
| Projected Gradient Descent | ▫️  |  ✔️  |     ➖      | Requires classifier derived from `LossGradientsMixin`                                          |
| Query-Efficient...         | ▪️  |  ➖   |     ➖      | from Wiki, but `QueryEfficientGradientEstimationClassifier` is not an attack implementation    |
| ShapeShifter Attack        | ▫️  |  ✔️  |     ➖      | Requires TensorFlowFasterRCNN classifier                                                       |
| SignOPT Attack             | ▪️  |  ➖   |     ➖      | This attack has not yet been tested for binary classification with a single output classifier  |
| SimBA                      | ▪️  |  ➖   |     ➖      | Requires neural network classifier                                                             |
| Spatial Transformation     | ▪️  |  ➖   |     ➖      | Requires neural network classifier                                                             |
| Square Attack              | ▪️  |  ➖   |     ➖      | Requires neural network classifier                                                             |
| Threshold Attack           | ▪️  |  ➖   |     ➖      | Requires neural network classifier                                                             |
| Zeroth Order Optimisation  | ▪️  |  ➖️  |     ✔️     |                                                                                                |
