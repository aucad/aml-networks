"""
Build a XGBoost model using IoT-23 data for provided dataset.

Provide as input a path to a dataset, or script uses default
dataset when none provided. Input data must use DMatrix format, see:
https://xgboost.readthedocs.io/en/stable/tutorials/input_format.html

Default usage:

```
python src/tree_xg.py
```

Specify input data to use:

```
python src/tree_xg.py ./path/to/train_data ./path/to/test_data
```
"""
import sys
from sys import argv

import xgboost as xgb

import tree_utils as tu

DEFAULT_DS = 'data/CTU-44-1.csv'


def train_boosted_tree(dataset=DEFAULT_DS, test_size=.1):
    """Train a classifier using XGBoost."""

    if test_size == 0:
        print('set test size > 0')
        sys.exit(1)

    attrs, classes, train_x, train_y, test_x, test_y = \
        tu.load_csv_data(dataset, test_size)

    dtrain = xgb.DMatrix(train_x, train_y)
    dtest = xgb.DMatrix(test_x, test_y)

    tu.show('Read dataset', dataset)
    tu.show('Attributes', len(attrs))
    tu.show('Classes', ", ".join([str(l) for l in classes]))
    tu.show('Training instances', dtrain.num_row())
    tu.show('Test instances', dtest.num_row())

    cls = xgb.train(
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
            'objective': 'binary:logistic'
        },
        # data to be trained
        dtrain=dtrain,
        # num_boost_round
        num_boost_round=2)

    # make prediction
    predictions = cls.predict(dtest)

    # score
    tu.score(test_y, predictions, display=True)

    return cls, attrs, dtrain, train_y, dtest, test_y


if __name__ == '__main__':
    ds = argv[1] if len(argv) > 1 else DEFAULT_DS
    train_boosted_tree(ds, 0.1)
