from sys import argv
from pathlib import Path
from os import path

from utility import read_csv

from sklearn import tree  # decision trees
from matplotlib import pyplot as plt  # save plots
from colorama import Fore, Style  # terminal colors

"""
This script builds a decision tree for provided dataset.
Provide as input a path to a dataset, or script uses default
dataset if none provided. The dataset must be numeric 
at all attributes.
"""

DEFAULT_DS = 'data/CTU-44-1.csv'
OUTPUT_DIR = 'adversarial'
DATASET_PATH = argv[1] if len(argv) > 1 else DEFAULT_DS
NAME = path.join(OUTPUT_DIR, Path(DATASET_PATH).stem)
ATTRS, ROWS = read_csv(DATASET_PATH)


def c(text):
    """color text"""
    return Fore.GREEN + str(text) + Style.RESET_ALL


def value_format(cell):
    """Numeric missing values have '?', replace with 0"""
    return 0 if cell == '?' else cell


def separate_labels(rows_, at_index=-1):
    """separate data from label.

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
    return new_rows, labels_, list(set(labels_))


def save_image(clf_, filename, feat_names, class_names, show):
    """Plot the tree and save to file."""
    plt.figure(dpi=200)
    tree.plot_tree(
        clf_, filled=True,
        feature_names=feat_names,
        class_names=class_names
    )
    plt.savefig(f'{filename}.png')
    if show:
        plt.show()


def train_tree(show_tree=False):
    """Train a decision tree"""
    print(f'Read dataset {c(NAME)}')
    print(f'Attributes: {c(len(ATTRS))}')
    print(f'Number of rows: {c(len(ROWS))}')

    X, y, classes = separate_labels(ROWS)
    clf = tree.DecisionTreeClassifier()
    clf.fit(X, y)
    save_image(clf, NAME, ATTRS, classes, show_tree)

    return clf, X, y


if __name__ == '__main__':
    train_tree(True)
