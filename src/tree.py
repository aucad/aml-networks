"""
This script builds a decision tree classifier for provided dataset.
Provide as input a path to a dataset, or script uses default
dataset if none provided. The dataset must be numeric
at all attributes.
"""

import warnings

warnings.filterwarnings("ignore")

from art.estimators.classification.scikitlearn \
    import ScikitlearnDecisionTreeClassifier as SkDT
from sklearn.tree import DecisionTreeClassifier, plot_tree

from src import AbsClassifierInstance


class DecisionTree(AbsClassifierInstance):

    def __init__(self, out):
        super().__init__('decision_tree', out)

    @staticmethod
    def formatter(x, y):
        return x

    def predict(self, data):
        return self.model.predict(data)

    def prep_model(self, robust):
        self.model = DecisionTreeClassifier()
        self.model.fit(self.train_x, self.train_y)

    def prep_classifier(self):
        self.classifier = SkDT(self.model)

    def tree_plotter(self):
        plot_tree(self.model, feature_names=self.attrs,
                  class_names=self.class_names, filled=True)
