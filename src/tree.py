"""
This script builds a decision tree classifier for provided dataset.
Provide as input a path to a dataset, or script uses default
dataset if none provided. The dataset must be numeric
at all attributes.

Default usage:

```
python src/tree.py
```

Usage with custom dataset:

```
python src/tree.py ./path/to/input_data.csv
```
"""
import logging
import warnings

warnings.filterwarnings("ignore")  # ignore import warnings

from sys import argv

from art.estimators.classification.scikitlearn \
    import ScikitlearnDecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn import tree

from abscls import AbsClassifierInstance

logger = logging.getLogger(__name__)


class DecisionTree(AbsClassifierInstance):

    def __init__(self):
        super().__init__('decision_tree')

    @staticmethod
    def formatter(x, y):
        return x

    def predict(self, data):
        return self.model.predict(data)

    def prep_model(self, robust):
        self.model = tree.DecisionTreeClassifier()
        self.model.fit(self.train_x, self.train_y)

    def prep_classifier(self):
        self.classifier = ScikitlearnDecisionTreeClassifier(self.model)

    def plot(self):
        """Plot the tree and save to file."""
        plt.figure(dpi=200)
        tree.plot_tree(
            self.model,
            feature_names=self.attrs,
            class_names=self.class_names,
            filled=True
        )
        plt.savefig(self.plot_path)
        plt.show()


if __name__ == '__main__':
    AbsClassifierInstance.default_run(DecisionTree, argv)
