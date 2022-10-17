# Adversarial attacks on tree-based classifiers

Link to notes:

<https://docs.google.com/document/d/1xqCDqrakSaXkOxijwse3zQ23Rm7-38m-rMJp7iLjS6A/edit?usp=sharing>

Descriptions of datasets:

<https://augustauniversity.box.com/s/ioj5v4ddiwcs9fs451rppzf5vhzohfpo>

## Setup

Required Python environment: 3.8 or 3.9 [^1]

[^1]: This is a hard requirement. Numpy requires >= 3.8; robust XGBoost relies on Python features removed after 3.9.

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

By default, attacks are configured to use a modified version of XGBoost classifier, enhanced with adversarial robustness
property.

The XGBoost classifier must be built locally [from source](./RobustTrees),
following [instructions here](./RobustTrees/tree/master/python-package#from-source).

After local build, set up Python environment to use this classifier version:

```
python -m pip install -e "/absolute/local/path/to/RobustTrees/python-package"
```

### Usage

This application has a command line interface. Standard interaction:

```
python -m src [ARGS]
```

To see available options, run:

```
python -m src --help
```

## Directory organization

| Directory           | Description                                                 |
|:--------------------|:------------------------------------------------------------|
| `results`           | Results of various experiments                              |
| 　`└─ comparisons`   | Comparison of classifiers, on IoT data, obtained using Weka |
| 　`└─ decision_tree` | Adversarial attacks on decision trees                       |
| 　`└─ xgboost`       | Adversarial attacks on XGBoost classifier                   |
| `data`              | Preprocessed data sets, ready for running various attacks   |
| `src`               | Source code                                                 |
| 　`└─ examples/`     | ART attack examples, for reference                          |
| 　`└─ attack.py`     | Abstract base class for an attack                           |
| 　`└─ cls.py`        | Abstract base class for a classifier                        |
| 　`└─ hopskip.py`    | HopSkipJump attack implementation                           |
| 　`└─ loader.py`     | Utility for loading models and attacks                      |
| 　`└─ tree.py`       | Decision tree classifier training                           |
| 　`└─ utility.py`    | Shared functionality and utils                              |
| 　`└─ validator.py`  | Post-attack record validator                                |
| 　`└─ xgb.py`        | XGBoost classifier training                                 |
| 　`└─ zoo.py`        | ZOO attack implementation                                   |

## Notes on ART Attacks

See [ART evasion attacks wiki ↗](https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/ART-Attacks#1-evasion-attacks)

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
| Query-efficient Black-box  | ▪️  |  ➖   |     ➖      | `QueryEfficientGradientEstimationClassifier` is not an attack implementation                   |
| ShapeShifter Attack        | ▫️  |  ✔️  |     ➖      | Requires TensorFlowFasterRCNN classifier                                                       |
| SignOPT Attack             | ▪️  |  ➖   |     ➖      | This attack has not yet been tested for binary classification with a single output classifier  |
| SimBA                      | ▪️  |  ➖   |     ➖      | Requires neural network classifier                                                             |
| Spatial Transformation     | ▪️  |  ➖   |     ➖      | Requires neural network classifier                                                             |
| Square Attack              | ▪️  |  ➖   |     ➖      | Requires neural network classifier                                                             |
| Threshold Attack           | ▪️  |  ➖   |     ➖      | Requires neural network classifier                                                             |
| Zeroth Order Optimisation  | ▪️  |  ➖️  |     ✔️     |                                                                                                |
