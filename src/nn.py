# flake8: noqa: E402

"""
This script builds a neural network classifier for provided dataset.
Provide as input a path to a dataset, or script uses default
dataset if none provided. The dataset must be numeric
at all attributes.
"""

import warnings

warnings.filterwarnings("ignore")

from art.estimators.classification.scikitlearn \
    import SklearnClassifier
from sklearn.neural_network import MLPClassifier

from src import Classifier


class NeuralNetwork(Classifier):

    def __init__(self, *args):
        super().__init__('neural_network', *args)

    @staticmethod
    def formatter(x, y):
        return x

    def predict(self, data):
        return self.model.predict(data)

    def prep_model(self, robust):
        self.model = MLPClassifier()
        self.model.fit(self.train_x, self.train_y)

    def prep_classifier(self):
        self.classifier = SklearnClassifier(self.model)

    def tree_plotter(self):
        pass
