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
import logging
from collections import Counter
from os import path
from pathlib import Path
from sys import argv

import xgboost as xgb
from art.estimators.classification import XGBoostClassifier
from matplotlib import pyplot as plt
from xgboost import plot_tree

import utility as tu

logger = logging.getLogger(__name__)


def formatter(x, y):
    return xgb.DMatrix(x, y)


def predict(model, data):
    tmp = model.predict(data)
    ax = 1 if len(tmp.shape) == 2 else 0
    return tmp.argmax(axis=ax)


def plot_tree_(clf_, filename="tree"):
    """Plot the tree and save to file."""
    plot_tree(clf_, num_trees=20, rankdir='LR')
    plt.tight_layout()
    plt.savefig(f'{filename}.png', dpi=200)
    plt.show()


def train(
        dataset=tu.DEFAULT_DS, test_size=.1, robust=False, max_size=-1,
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
        tu.load_csv_data(dataset, test_size, max_size=max_size)

    dtrain = formatter(train_x, train_y)
    evallist = [(dtrain, 'eval'), (dtrain, 'train')]
    cl_n, cl_srtd = len(classes), sorted(classes)
    cl_frq = Counter(train_y)
    cl_names = [tu.text_label(cl) for cl in cl_srtd]
    cl_ratio = [f'{(100 * cl_frq[k] / len(train_y)):.1f}'
                for k in cl_srtd]

    model = xgb.train(
        # Booster params
        # https://xgboost.readthedocs.io/en/stable/parameter.html
        params={
            # tree_method controls which training method to use.
            # We add a new option robust_exact for this parameter.
            # Setting tree_method = robust_exact will use our proposed
            # robust training. For other training methods, please refer
            # to XGBoost documentation.
            'tree_method': 'robust_exact' if robust else 'exact',
            # set XGBoost to do multiclass classification using the
            # softmax objective,
            # multi:softprob: outputs a vector of ndata * nclass,
            # which can be further reshaped to ndata * nclass matrix.
            # The result contains predicted probability of each data
            # point belonging to each class.
            # A default metric will be assigned according to objective.
            'objective': 'multi:softprob',
            # multi:softprob: requires setting num_class(number of classes).
            'num_class': cl_n,
            # silence most console output
            'verbose_eval': False,
            'silent': True,
        },
        num_boost_round=20,
        dtrain=dtrain,
        evals=evallist)

    tu.show('Read dataset', dataset)
    tu.show('Attributes', len(attrs))
    tu.show('Classifier', f'XGBoost, robust: {robust}')
    tu.show('Classes', ", ".join(cl_names))
    tu.show('Class split', "/".join(cl_ratio))
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
    name = path.join('results/xgboost', Path(ds).stem)
    train(ds, 0.05, plot=False, fn=name)
