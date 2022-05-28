"""
This script builds a decision tree for provided dataset.
Provide as input a path to a dataset, or script uses default
dataset if none provided. The dataset must be numeric
at all attributes.

Default usage:

```
python src/tree.py
```


Usage with custom dataset:

```
python src/tree.py ./path/to/input_data.csv
```
"""

from os import path
from pathlib import Path
from sys import argv

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split

from utility import color_text as c

DEFAULT_DS = 'data/CTU-44-1.csv'


def text_label(i):
    return 'benign' if i == 1 else 'malicious'


def int_label(text):
    return 1 if text.lower() == 'benign' else 0


def show(msg, value):
    print(f'{msg} '.ljust(30, '-'), c(value))


def save_image(clf_, feat_names, class_names, filename):
    """Plot the tree and save to file."""
    plt.figure(dpi=200)
    tree.plot_tree(
        clf_, filled=True,
        feature_names=feat_names,
        class_names=[text_label(cn) for cn in class_names]
    )
    plt.savefig(f'{filename}.png')
    plt.show()


def normalize(data):
    """normalize values in range 0.0 - 1.0."""
    for i in range(len(data[0])):
        data[:, i] = (data[:, i]) / max(data[:, i])
        data[:, i] = np.nan_to_num(data[:, i])
    return data


def train_tree(ds=DEFAULT_DS, plot=False, test_size=0.1, name="tree"):
    """Train a decision tree"""

    df = pd.read_csv(ds)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    attrs = [col for col in df.columns]
    split = 0 < test_size < len(df)
    test_x, test_y = [], []

    # sample training/test instances
    if split:
        train, test = train_test_split(df, test_size=test_size)
        test_x = normalize(np.array(test)[:, :-1])
        test_y = np.array(test)[:, -1].astype(int).flatten()
    else:
        train = df

    train_x = normalize(np.array(train)[:, :-1])
    train_y = np.array(train)[:, -1].astype(int).flatten()
    classes = np.unique(train_y)

    clf = tree.DecisionTreeClassifier()
    clf.fit(train_x, train_y)

    show('Read dataset', ds)
    show('Attributes', len(attrs))
    show('Classes', ", ".join([str(l) for l in classes]))
    show('Training instances', len(train_x))
    show('Test instances', len(test_x))

    if plot:
        save_image(clf, attrs, classes, name)

    if split:
        acc = clf.score(test_x, test_y)
        split = [f'{v} ({np.count_nonzero(test_y == v)})'
                 for v in classes]
        show('Split for score', " | ".join(split))
        show('Score', f'{acc * 100:.2f} %')

    return clf, attrs, train_x, train_y, test_x, test_y


if __name__ == '__main__':
    ds = argv[1] if len(argv) > 1 else DEFAULT_DS
    name = path.join('adversarial', Path(ds).stem)
    train_tree(ds, True, 0.0, name)
