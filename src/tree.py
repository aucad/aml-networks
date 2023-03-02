"""
Train a decision tree.
"""

import warnings

warnings.filterwarnings("ignore")

from art.estimators.classification.scikitlearn \
    import ScikitlearnDecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

from src import Classifier


class DecisionTree(Classifier):

    def __init__(self, *args):
        super().__init__('decision_tree', *args)

    @staticmethod
    def formatter(x, y):
        return x

    def predict(self, data):
        return self.model.predict(data)

    def prep_model(self, robust):
        self.model = DecisionTreeClassifier()
        self.model.fit(self.train_x, self.train_y)

    def prep_classifier(self):
        self.classifier = ScikitlearnDecisionTreeClassifier(self.model)

    def tree_plotter(self):
        plot_tree(self.model, feature_names=self.attrs,
                  class_names=self.class_names, filled=True)
