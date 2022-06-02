"""
Build a XGBoost model using IoT-23 data for provided dataset.

Provide as input a path to a dataset, or script uses default
dataset if none provided. The dataset must be numeric at all attributes.

Default usage:

```
python src/tree_xg.py
```

Specify input data to use:

```
python src/tree_xg.py ./path/to/input_data.csv
```
"""
from sys import argv

import numpy as np
import xgboost as xgb

import tree_utils as tu
from tree_utils import DEFAULT_DS

from art.estimators.classification import XGBoostClassifier


def formatter(x, y):
    return xgb.DMatrix(x, y)


def binarize(values):
    return np.array([int(round(val, 0)) for val in values]) \
        .astype(int).flatten()


def train_tree(dataset=DEFAULT_DS, test_size=.1):
    """Train a classifier using XGBoost."""

    attrs, classes, train_x, train_y, test_x, test_y = \
        tu.load_csv_data(dataset, test_size)

    dtrain = formatter(train_x, train_y)

    model = xgb.train(
        # Booster params
        # https://xgboost.readthedocs.io/en/stable/parameter.html
        params={
            # Maximum depth of a tree. Increasing this value will
            # make the model more complex and more likely to overfit.
            # 0 indicates no limit on depth. Beware that XGBoost
            # aggressively consumes memory when training a deep tree.
            # exact tree method requires non-zero value.
            'max_depth': 2,
            # Evaluation metrics for validation data, a default
            # metric will be assigned according to objective (rmse
            # for regression, and logloss for classification,
            # mean average precision for ranking) User can add
            # multiple evaluation metrics. Remember to pass the
            # metrics in as list of parameters pairs instead of map,
            # so that latter eval_metric won’t override previous one
            'eval_metric': 'error',
            # Step size shrinkage used in update to prevents
            # overfitting. After each boosting step, we can directly
            # get the weights of new features, and eta shrinks the
            # feature weights to make the boosting process more
            # conservative. (range: [0,1])
            'eta': 1,
            # the learning task and the corresponding learning
            # - objective binary:logistic -> logistic regression for
            #   binary classification, output probability
            'objective': 'binary:logistic',
        },
        dtrain=dtrain,
        num_boost_round=10)

    split = np.count_nonzero(test_y == 1)

    tu.show('Read dataset', dataset)
    tu.show('Attributes', len(attrs))
    tu.show('Classes', ", ".join([tu.text_label(l) for l in classes]))
    tu.show('Training instances', dtrain.num_row())
    tu.show('Test instances', len(test_x))
    tu.show('Split (benign)', f'{100 * split / len(train_y):.2f} %')

    # evaluate performance
    if len(test_x) > 0:
        dtest = formatter(test_x, test_y)
        predictions = binarize(model.predict(dtest))
        tu.score(test_y, predictions, display=True)
    else:
        predictions = binarize(model.predict(dtrain))
        tu.score(train_y, predictions, display=True)

    cls = XGBoostClassifier(
        model=model,
        # clip_values=(0.0, 1.0),
        nb_features=len(train_x[0]),
        nb_classes=len(classes))

    return cls, model, attrs, train_x, train_y, test_x, test_y


if __name__ == '__main__':
    ds = argv[1] if len(argv) > 1 else DEFAULT_DS
    train_tree(ds, 0)
