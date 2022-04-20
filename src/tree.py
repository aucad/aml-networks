"""
This script builds a decision tree for provided dataset.
Provide as input a path to a dataset, or script uses default
dataset if none provided. The dataset must be numeric
at all attributes.


Usage:

```
python src/tree.py
```
"""

from os import path
from pathlib import Path
from sys import argv

from matplotlib import pyplot as plt
from sklearn import tree

from utility import read_csv, color_text as c

DEFAULT_DS = 'data/CTU-44-1.csv'
OUTPUT_DIR = 'adversarial'
DATASET_PATH = argv[1] if len(argv) > 1 else DEFAULT_DS
NAME = path.join(OUTPUT_DIR, Path(DATASET_PATH).stem)
ATTRS, ROWS = read_csv(DATASET_PATH)


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
        new_rows.append([value_format(value) for value in row])
    labels_ = [1 if l.lower() == 'benign' else 0 for l in labels_]
    return new_rows, labels_, list(set(labels_))


def save_image(clf_, filename, feat_names, class_names):
    """Plot the tree and save to file."""
    plt.figure(dpi=200)
    tree.plot_tree(
        clf_, filled=True,
        feature_names=feat_names,
        class_names=class_names
    )
    plt.savefig(f'{filename}.png')
    plt.show()


def train_tree(show_tree=False):
    """Train a decision tree"""
    print(f'Read dataset {c(NAME)}')
    print(f'Attributes: {c(len(ATTRS))}')
    print(f'Number of rows: {c(len(ROWS))}')

    X, y, classes = separate_labels(ROWS)
    clf = tree.DecisionTreeClassifier()
    clf.fit(X, y)
    if show_tree:
        save_image(clf, NAME, ATTRS, classes)

    return clf, X, y


if __name__ == '__main__':
    train_tree(True)
