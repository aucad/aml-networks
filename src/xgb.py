"""
Train XGBoost classifier.

This is a custom version of standard XGBoost with added robustness.

* The base model should be locally built from "RobustTrees":
  <https://github.com/chenhongge/RobustTrees>

* The instructions for building from source are provided here:
  <https://github.com/chenhongge/RobustTrees/tree/master/python-package#from-source>
"""
import os
import sys

from art.estimators.classification import XGBoostClassifier

# noinspection PyPackageRequirements
from xgboost import DMatrix, train as xg_train

from src import Classifier


class XgBoost(Classifier):

    def __init__(self, *args):
        super().__init__('xgboost', *args)

    @staticmethod
    def formatter(x, y):
        return DMatrix(x, y)

    def predict(self, data):
        tmp = self.model.predict(data)
        ax = 1 if len(tmp.shape) == 2 else 0
        return tmp.argmax(axis=ax)

    def init_learner(self, robust: bool) -> None:
        """
        Full list of options:
        <https://xgboost.readthedocs.io/en/stable/parameter.html>
        """
        d_train = self.formatter(self.train_x, self.train_y)

        # Hack to _actually_ block print to console,
        # thanks to https://stackoverflow.com/a/8391735
        sys.stdout = open(os.devnull, 'w')

        self.model = xg_train(
            num_boost_round=20,
            dtrain=d_train,
            evals=[(d_train, 'eval'), (d_train, 'train')],
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
                **self.cls_conf
            })
        sys.stdout = sys.__stdout__  # re-enable print

        self.classifier = XGBoostClassifier(
            model=self.model,
            nb_features=self.n_features,
            nb_classes=self.n_classes,
            clip_values=(0, 1))
