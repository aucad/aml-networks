"""
This script builds a decision tree for provided dataset.
Provide as input a path to a dataset, or script uses default
dataset if none provided. The dataset must be numeric
at all attributes.


Usage:

```
python src/tree.py ./path/to/input_data.csv
```
"""

from os import path
from pathlib import Path
from sys import argv
from random import sample

import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree

from utility import read_csv, color_text as c

DEFAULT_DS = 'data/CTU-44-1.csv'
OUTPUT_DIR = 'adversarial'
DATASET_PATH = argv[1] if len(argv) > 1 else DEFAULT_DS
NAME = path.join(OUTPUT_DIR, Path(DATASET_PATH).stem)
ATTRS, ROWS = read_csv(DATASET_PATH)


def text_label(i):
    return 'benign' if i == 1 else 'malicious'


def int_label(text):
    return 1 if text.lower() == 'benign' else 0


def value_format(cell):
    """Numeric missing values have '?', replace with 0"""
    return 0 if cell == '?' else (
        float(cell) if "." in cell else int(cell))


def separate_labels(rows_, at_index=-1):
    """Separates data from class label.

    Arguments:
        rows_: list of data rows with labels
        at_index: label index in row vector (default: last index)

    Returns:
        Two lists where first contains data vectors, second
        contains the labels (order preserved).
    """
    new_rows, labels_ = [], []
    for row in rows_:
        labels_.append(row.pop(at_index))
        new_rows.append(
            [value_format(value) for value in row])
    labels_ = [int_label(l) for l in labels_]
    return new_rows, labels_, list(set(labels_))


def format_data(num_classes, x_train, y_train):
    # use numpy arrays
    x_train = np.array([np.array(xi) for xi in x_train])
    y_train = np.array(y_train)

    x_train = x_train[y_train < num_classes][:, :]
    y_train = y_train[y_train < num_classes]

    # normalize all attributes to range 0.0 - 1.0
    for i in range(len(x_train[0])):
        x_train[:, i] = (x_train[:, i]) / max(x_train[:, i])

    return x_train, y_train


def save_image(clf_, filename, feat_names, class_names):
    """Plot the tree and save to file."""
    plt.figure(dpi=200)
    tree.plot_tree(
        clf_, filled=True,
        feature_names=feat_names,
        class_names=[str(cn) for cn in class_names]
    )
    plt.savefig(f'{filename}.png')
    plt.show()


def sample_test(x, y, test_set):
    sample_size = int(len(x) * test_set)
    idx = sample(list(range(len(x))), sample_size)

    train_x = np.array([x_ for i, x_ in enumerate(x) if i not in idx])
    train_y = np.array([y_ for i, y_ in enumerate(y) if i not in idx])
    test_x = np.array([x_ for i, x_ in enumerate(x) if i in idx])
    test_y = np.array([y_ for i, y_ in enumerate(y) if i in idx])
    return (train_x, train_y), (test_x, test_y)


def train_tree(show_tree=False, test_set=0):
    """Train a decision tree"""
    print(f'Read dataset: {c(NAME)}')
    print(f'Attributes:   {c(len(ATTRS))}')
    print(f'Instances:    {c(len(ROWS))}')

    x_, y_, classes = separate_labels(ROWS)
    x, y = format_data(len(classes), x_, y_)
    (train_x, train_y), (test_x, test_y) = sample_test(x, y, test_set)
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_x, train_y)
    if show_tree:
        save_image(clf, NAME, ATTRS, classes)

    return clf, train_x, train_y, ATTRS, test_x, test_y


if __name__ == '__main__':
    train_tree(True)
