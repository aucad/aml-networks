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

from sys import argv

import xgboost as xgb

from utility import color_text as c

OUTPUT_DIR = 'boosted'
DEFAULT_TRAIN = 'data/CTU-44-1.train.dmat'
DEFAULT_TEST = 'data/CTU-44-1.test.dmat'
TRAIN_PATH = argv[1] if len(argv) > 1 else DEFAULT_TRAIN
TEST_PATH = argv[2] if len(argv) > 2 else DEFAULT_TEST


def score(test_labels, predictions):
    """0 is malicious --> positive"""
    accuracy, precision, recall, f_score = 0, 0, 0, 0
    tp, tn, fn, fp, tot = 0, 0, 0, 0, len(test_labels)
    for actual, pred in zip(test_labels, predictions):
        int_pred = int(round(pred, 0))
        tp += 1 if actual == int_pred and int_pred == 0 else 0
        tn += 1 if actual == int_pred and int_pred == 1 else 0
        fp += 1 if actual != int_pred and int_pred == 0 else 0
        fn += 1 if actual != int_pred and int_pred == 1 else 0
    if tot > 0:
        accuracy = (tp + tn) / tot
    if tp > 0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f_score = tp / (tp + 0.5 * (fp + fn))
    return accuracy, precision, recall, f_score


def load_data(train_path, test_path):
    print(f'Reading: {c(train_path)} and {c(test_path)}')

    # read in data
    dtrain = xgb.DMatrix(train_path)
    dtest = xgb.DMatrix(test_path)

    # display numbers
    print(f"Training data size: {c(dtrain.num_row())}")
    print(f"Test data size: {c(dtest.num_row())}")
    return dtrain, dtest


def train_boosted_tree():
    """Train a classifier using XGBoost."""

    num_rounds = 2
    dtrain, dtest = load_data(TRAIN_PATH, TEST_PATH)
    dtrain_y, dtest_y = dtrain.get_label(), dtest.get_label()

    # specify parameters via map
    param = {'max_depth': 2, 'eval_metric': 'auc',
             'eta': 1, 'objective': 'binary:logistic'}
    clf = xgb.train(param, dtrain, num_rounds)

    # make prediction
    preds = clf.predict(dtest)

    # score
    acc, pre, rec, fs = score(dtest_y, preds)

    print('Accuracy ----- ', c(f'{acc * 100:.2f} %'))
    print('Precision ---- ', c(f'{pre * 100:.2f} %'))
    print('Recall ------- ', c(f'{rec * 100:.2f} %'))
    print('F-score ------ ', c(f'{fs * 100:.2f} %'))

    return clf, dtrain, dtrain_y, None, dtest, dtest_y


if __name__ == '__main__':
    train_boosted_tree()
