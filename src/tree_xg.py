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

import xgboost as xgb

import tree_utils as tu
from tree_utils import DEFAULT_DS

from art.estimators.classification import XGBoostClassifier


def formatter(x, y):
    return xgb.DMatrix(x, y)


def predict(model, data):
    tmp = model.predict(data)
    return tmp.argmax(axis=1)


def train_tree(dataset=DEFAULT_DS, test_size=.1):
    """Train a classifier using XGBoost."""

    attrs, classes, train_x, train_y, test_x, test_y = \
        tu.load_csv_data(dataset, test_size)

    dtrain = formatter(train_x, train_y)
    evallist = [(dtrain, 'eval'), (dtrain, 'train')]

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
            'objective': 'multi:softprob',
            # from example ??
            'metric': 'multi_logloss',
            'num_class': len(classes)
        },
        dtrain=dtrain,
        num_boost_round=10,
        evals=evallist)

    tu.show('Read dataset', dataset)
    tu.show('Attributes', len(attrs))
    tu.show('Classes', ", ".join([tu.text_label(l) for l in classes]))
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

    cls = XGBoostClassifier(
        model=model,
        clip_values=(0, 1),
        nb_features=len(train_x[0]),
        nb_classes=len(classes))

    return cls, model, attrs, train_x, train_y, test_x, test_y


if __name__ == '__main__':
    ds = argv[1] if len(argv) > 1 else DEFAULT_DS
    train_tree(ds, 0.05)
