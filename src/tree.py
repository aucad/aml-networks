"""
A decision tree classifier implementation.

This is a basic wrapper for the non-robust scikit-learn decision tree
classifier.
"""

import warnings

warnings.filterwarnings("ignore")

from art.estimators.classification.scikitlearn \
    import ScikitlearnDecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

from src import Classifier


class DecisionTree(Classifier):

    def __init__(self, *args):
        super().__init__('decision_tree', *args)

    @staticmethod
    def formatter(x, y):
        return x

    def predict(self, data):
        return self.model.predict(data)

    def init_learner(self, robust):
        self.model = DecisionTreeClassifier(
            # Measures quality of a split
            # "gini" or "entropy"
            criterion="gini",
            # Strategy used to choose the split at each node
            # "random" or "best"
            splitter="random",
            # Maximum depth of the tree
            max_depth=None,
            # Minimum number of samples required to split an internal node
            min_samples_split=2,
            # Minimum number of samples required to be at a leaf node
            min_samples_leaf=1,
            # The minimum weighted fraction of the sum total of weights (of all
            # the input samples) required to be at a leaf node
            min_weight_fraction_leaf=0.0,
            # Number of features to consider when looking for the best split
            # "auto", "sqrt", "log2"
            max_features="auto",
            # Used to pick randomly the `max_features` used at each split
            random_state=None,
            # Grow a tree with `max_leaf_nodes` in best-first fashion
            max_leaf_nodes=None,
            # Node will be split if it induces a decrease of the impurity
            min_impurity_decrease=0.0,
            # Weights associated with classes in form `{class_label: weight}`.
            # If None, all classes are supposed to have weight one.
            class_weight=None,
            # Complexity parameter used for Minimal Cost-Complexity Pruning
            ccp_alpha=0.0
        )

        self.model.fit(self.train_x, self.train_y)
        self.classifier = ScikitlearnDecisionTreeClassifier(self.model)
