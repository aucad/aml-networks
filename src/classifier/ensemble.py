"""
Build a XGBoost model for provided dataset.

Provide as input a path to a dataset, or script uses default
dataset if none provided. The dataset must be numeric at all attributes.

Default usage:

```
python src/classifier/ensemble.py
```

Specify input data to use:

```
python src/classifier/ensemble.py ./path/to/input_data.csv
```
"""
import logging
from sys import argv

from art.estimators.classification import XGBoostClassifier
from matplotlib import pyplot as plt
# noinspection PyPackageRequirements
from xgboost import plot_tree, DMatrix, train as xg_train

from base import AbsClassifierInstance

logger = logging.getLogger(__name__)


class XGBClassifier(AbsClassifierInstance):

    def __init__(self):
        super().__init__('gxboost')

    @staticmethod
    def formatter(x, y):
        return DMatrix(x, y)

    def predict(self, data):
        tmp = self.model.predict(data)
        ax = 1 if len(tmp.shape) == 2 else 0
        return tmp.argmax(axis=ax)

    def plot(self):
        """Plot the tree and save to file."""
        plot_tree(self.classifier, num_trees=20, rankdir='LR')
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=200)
        plt.show()

    def prep_model(self, robust):
        """
        see:  https://xgboost.readthedocs.io/en/stable/parameter.html
        for full list of options
        """
        self.model = xg_train(
            num_boost_round=20,
            params={
                # tree_method controls which training method to use.
                # We add a new option robust_exact for this
                # parameter. Setting tree_method = robust_exact will
                # use our proposed robust training. For other
                # training methods, please refer to XGBoost
                # documentation.
                'tree_method': 'robust_exact' if robust else 'exact',
                # set XGBoost to do multiclass classification using
                # the softmax objective, multi:softprob: outputs a
                # vector of ndata * nclass, which can be further
                # reshaped to ndata * nclass matrix. The result
                # contains predicted probability of each data point
                # belonging to each class. A default metric will be
                # assigned according to objective.
                'objective': 'multi:softprob',
                # multi:softprob: requires setting num_class
                'num_class': self.n_classes,
                # try silence most console output
                'verbose_eval': False,
                'silent': True,
            },
            dtrain=(dtrain := self.formatter(
                self.train_x, self.train_y)),
            evals=[(dtrain, 'eval'), (dtrain, 'train')])

    def prep_classifier(self):
        self.classifier = XGBoostClassifier(
            model=self.model,
            nb_features=self.n_features,
            nb_classes=self.n_classes,
            clip_values=(0, 1))


if __name__ == '__main__':
    AbsClassifierInstance.default_run(XGBClassifier, argv)
