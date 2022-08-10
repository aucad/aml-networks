"""
This script builds a decision tree classifier for provided dataset.
Provide as input a path to a dataset, or script uses default
dataset if none provided. The dataset must be numeric
at all attributes.

Default usage:

```
python src/train_dt.py
```


Usage with custom dataset:

```
python src/train_dt.py ./path/to/input_data.csv
```
"""
import warnings

warnings.filterwarnings("ignore")  # ignore import warnings

from os import path
from pathlib import Path
from sys import argv

import numpy as np
from art.estimators.classification.scikitlearn import \
    ScikitlearnDecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn import tree

import utility as tu

formatter = None


def predict(model, data):
    return model.predict(data)


def plot_tree(clf_, feat_names, class_names, filename="tree"):
    """Plot the tree and save to file."""
    plt.figure(dpi=200)
    tree.plot_tree(
        clf_, filled=True,
        feature_names=feat_names,
        class_names=[tu.text_label(cn) for cn in class_names]
    )
    plt.savefig(f'{filename}.png')
    plt.show()


def train(dataset=tu.DEFAULT_DS, test_size=.1, plot=False, fn=None):
    """Train a decision tree"""

    attrs, classes, train_x, train_y, test_x, test_y = \
        tu.load_csv_data(dataset, test_size)

    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)

    tu.show('Read dataset', dataset)
    tu.show('Attributes', len(attrs))
    tu.show('Classifier', 'decision tree')
    tu.show('Classes', ", ".join([tu.text_label(l) for l in classes]))
    tu.show('Training instances', len(train_x))
    tu.show('Test instances', len(test_x))

    if len(test_x) > 0:
        predictions = predict(model, test_x)
        split = [str(np.count_nonzero(test_y == v)) for v in classes]
        tu.show('Test split', "/".join(split))
        tu.score(test_y, predictions, 0, display=True)

    if plot:
        plot_tree(model, attrs, classes, fn)

    classifier = ScikitlearnDecisionTreeClassifier(model)

    return classifier, model, attrs, train_x, train_y, test_x, test_y


if __name__ == '__main__':
    ds = argv[1] if len(argv) > 1 else tu.DEFAULT_DS
    name = path.join('adversarial_dt', Path(ds).stem)
    train(ds, 0.2, False, name)
