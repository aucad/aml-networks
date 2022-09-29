"""
This script builds a decision tree classifier for provided dataset.
Provide as input a path to a dataset, or script uses default
dataset if none provided. The dataset must be numeric
at all attributes.
"""

import warnings

warnings.filterwarnings("ignore")

from art.estimators.classification.scikitlearn \
    import ScikitlearnDecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

from cls import AbsClassifierInstance


class DecisionTree(AbsClassifierInstance):

    def __init__(self):
        super().__init__('decision_tree')

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

    def plot(self):
        """Plot the tree and save to file."""
        plt.figure(dpi=200)
        plot_tree(
            self.model,
            feature_names=self.attrs,
            class_names=self.class_names,
            filled=True
        )
        plt.savefig(self.plot_path)
        plt.show()


if __name__ == '__main__':
    AbsClassifierInstance.default_run(DecisionTree)
