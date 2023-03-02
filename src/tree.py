"""
Train a decision tree.
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
        self.model = DecisionTreeClassifier()
        self.model.fit(self.train_x, self.train_y)
        self.classifier = ScikitlearnDecisionTreeClassifier(self.model)
