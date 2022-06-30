"""
Build a XGBoost model using IoT-23 data for provided dataset.

Provide as input a path to a dataset, or script uses default
dataset if none provided. The dataset must be numeric at all attributes.

Default usage:

```
python src/train_xg.py
```

Specify input data to use:

```
python src/train_xg.py ./path/to/input_data.csv
```
"""

from os import path
from pathlib import Path
from sys import argv

import xgboost as xgb
from art.estimators.classification import XGBoostClassifier
from matplotlib import pyplot as plt
from xgboost import plot_tree

import utility as tu


def formatter(x, y):
    return xgb.DMatrix(x, y)


def predict(model, data):
    tmp = model.predict(data)
    return tmp.argmax(axis=1)


def plot_tree_(clf_, filename="tree"):
    """Plot the tree and save to file."""
    plot_tree(clf_, num_trees=20, rankdir='LR')
    plt.tight_layout()
    plt.savefig(f'{filename}.png', dpi=200)
    plt.show()


def train(
        dataset=tu.DEFAULT_DS, test_size=.1, robust=False, max=-1,
        plot=False, fn=None
):
    """Train a classifier using XGBoost.

    Arguments:
        dataset - path to dataset
        test_size - 0.0 < 1.0 percentage split for test set
        robust - set to True to use robust training
        max - max cap for number of training instances
    """

    attrs, classes, train_x, train_y, test_x, test_y = \
        tu.load_csv_data(dataset, test_size, max=max)

    dtrain = formatter(train_x, train_y)
    evallist = [(dtrain, 'eval'), (dtrain, 'train')]
    pos = 100 * sum([1 for y in train_y if y == 1]) / len(train_y)

    model = xgb.train(
        # Booster params
        # https://xgboost.readthedocs.io/en/stable/parameter.html
        params={
            # set XGBoost to do multiclass classification using the
            # softmax objective, you also need to set num_class(
            # number of classes)
            # multi:softprob: outputs a vector of ndata * nclass,
            # which can be further reshaped to ndata * nclass matrix.
            # The result contains predicted probability of each data
            # point belonging to each class.
            'tree_method': 'robust_exact' if robust else 'exact',
            'objective': 'multi:softprob',
            'metric': 'multi_logloss',
            'num_class': len(classes),
            'verbosity': 0,
        },
        num_boost_round=20,
        dtrain=dtrain,
        evals=evallist)

    tu.show('Read dataset', dataset)
    tu.show('Attributes', len(attrs))
    tu.show('Classifier', f'XGBoost, robust: {robust}')
    tu.show('Classes', ", ".join([tu.text_label(l) for l in classes]))
    tu.show('Training split', f'{pos:.1f}/{(100 - pos):.1f}')
    tu.show('Training instances', dtrain.num_row())
    tu.show('Test instances', len(test_x))

    # evaluate performance
    if len(test_x) > 0:
        dtest = xgb.DMatrix(test_x, test_y)
        predictions = predict(model, dtest)
        tu.score(test_y, predictions, display=True)
    else:
        predictions = predict(model, dtrain)
        tu.score(train_y, predictions, display=True)

    if plot:
        plot_tree_(model, fn)

    cls = XGBoostClassifier(
        model=model,
        clip_values=(0, 1),
        nb_features=len(train_x[0]),
        nb_classes=len(classes))

    return cls, model, attrs, train_x, train_y, test_x, test_y


if __name__ == '__main__':
    ds = argv[1] if len(argv) > 1 else tu.DEFAULT_DS
    name = path.join('adversarial_xg', Path(ds).stem)
    train(ds, 0.05, plot=False, fn=name)
