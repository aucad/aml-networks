"""
This script builds a decision tree classifier for provided dataset.
Provide as input a path to a dataset, or script uses default
dataset if none provided. The dataset must be numeric
at all attributes.

Default usage:

```
python src/train_dt.py
```


Usage with custom dataset:

```
python src/train_dt.py ./path/to/input_data.csv
```
"""
import logging
import warnings

import numpy as np

warnings.filterwarnings("ignore")  # ignore import warnings

from os import path
from sys import argv

from art.estimators.classification.scikitlearn import \
    ScikitlearnDecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn import tree

from train_cls import AbsClassifierInstance
import utility as tu

logger = logging.getLogger(__name__)


class DecisionTree(AbsClassifierInstance):

    def __init__(self):
        super().__init__('decision_tree')

    @staticmethod
    def formatter(x, y):
        return x

    def predict(self, data):
        return self.model.predict(data)

    def plot(self):
        """Plot the tree and save to file."""
        plt.figure(dpi=200)
        tree.plot_tree(
            self.model,
            feature_names=self.attrs,
            class_names=self.class_names,
            filled=True
        )
        plt.savefig(path.join(tu.RESULT_DIR, self.plot_filename))
        plt.show()

    def train(self, dataset, test_percent):
        self.ds_path = dataset
        self.test_percent = test_percent

        attrs, classes, train_x, train_y, test_x, test_y = \
            tu.load_csv_data(dataset, self.test_percent)

        self.attrs = attrs
        self.classes = classes
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

        self.model = tree.DecisionTreeClassifier()
        self.model.fit(train_x, train_y)
        self.classifier = ScikitlearnDecisionTreeClassifier(self.model)
        self.train_stats()

        if self.test_size > 0:
            predictions = self.predict(self.test_x)
            split = [str(np.count_nonzero(self.test_y == v))
                     for v in self.classes]
            tu.show('Test split', "/".join(split))
            tu.score(self.test_y, predictions, 0, display=True)

        return self.classifier, self.model, attrs, train_x, \
               train_y, test_x, test_y


DT = DecisionTree()

formatter = DT.formatter


def predict(model, data):
    return DT.predict(data)


def train(dataset=tu.DEFAULT_DS, test_size=.1, plot=False, fn=None):
    return DT.train(dataset, test_size)


if __name__ == '__main__':
    ds = argv[1] if len(argv) > 1 else tu.DEFAULT_DS
    DT.train(ds, 0.2)
