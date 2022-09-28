"""
Build a XGBoost model for provided dataset.

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
from os import path
from sys import argv

import xgboost as xgb
from art.estimators.classification import XGBoostClassifier
from matplotlib import pyplot as plt
from xgboost import plot_tree

from train_cls import AbsClassifierInstance
import utility as tu

logger = logging.getLogger(__name__)


class XGBClassifier(AbsClassifierInstance):

    def __init__(self, dataset_path):
        super().__init__('gxboost', dataset_path)

    @staticmethod
    def formatter(x, y):
        return xgb.DMatrix(x, y)

    def predict(self, data):
        tmp = self.model.predict(data)
        ax = 1 if len(tmp.shape) == 2 else 0
        return tmp.argmax(axis=ax)

    def plot(self):
        """Plot the tree and save to file."""
        plot_tree(self.classifier, num_trees=20, rankdir='LR')
        plt.tight_layout()
        fn = path.join(tu.RESULT_DIR, self.plot_filename)
        plt.savefig(fn, dpi=200)
        plt.show()

    def train(self, test_percent, robust=False):
        """Train a classifier using XGBoost.

        Arguments:
            dataset - path to dataset
            test_size - 0.0 < 1.0 percentage split for test set
            robust - set to True to use robust training
            max - max cap for number of training instances
        """
        self.test_percent = test_percent

        attrs, classes, train_x, train_y, test_x, test_y = \
            tu.load_csv_data(self.ds_path, test_percent)

        self.attrs = attrs
        self.classes = classes
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

        dtrain = self.formatter(train_x, train_y)
        evallist = [(dtrain, 'eval'), (dtrain, 'train')]
        cl_n, cl_srtd = len(classes), sorted(classes)

        self.model = xgb.train(
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

        self.train_stats()

        # evaluate performance
        if len(test_x) > 0:
            dtest = xgb.DMatrix(test_x, test_y)
            predictions = self.predict(dtest)
            tu.score(test_y, predictions, display=True)
        else:
            predictions = self.predict(dtrain)
            tu.score(train_y, predictions, display=True)

        self.classifier = XGBoostClassifier(
            model=self.model,
            clip_values=(0, 1),
            nb_features=len(train_x[0]),
            nb_classes=len(classes))

        return self.classifier, self.model, attrs, train_x, \
               train_y, test_x, test_y


if __name__ == '__main__':
    ds = argv[1] if len(argv) > 1 else tu.DEFAULT_DS
    XGBClassifier(ds).train(0.05)
