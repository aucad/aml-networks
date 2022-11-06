"""
Build a XGBoost model for provided dataset.

Provide as input a path to a dataset, or script uses default
dataset if none provided. The dataset must be numeric at all attributes.
"""
import os
import sys

from art.estimators.classification import XGBoostClassifier
# noinspection PyPackageRequirements
from xgboost import plot_tree, DMatrix, train as xg_train

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

    # noinspection PyBroadException
    def tree_plotter(self):
        """Plot the tree and save to file."""
        plot_tree(self.model, num_trees=20, rankdir='LR')

    def prep_model(self, robust):
        """
        see:  https://xgboost.readthedocs.io/en/stable/parameter.html
        for full list of options
        """
        d_train = self.formatter(self.train_x, self.train_y)
        # https://stackoverflow.com/a/8391735
        sys.stdout = open(os.devnull, 'w')  # block print
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
            })
        sys.stdout = sys.__stdout__  # re-enable print

    def prep_classifier(self):
        self.classifier = XGBoostClassifier(
            model=self.model,
            nb_features=self.n_features,
            nb_classes=self.n_classes,
            clip_values=(0, 1))
