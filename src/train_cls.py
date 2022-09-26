import logging
from pathlib import Path

import numpy as np

import utility as tu

logger = logging.getLogger(__name__)


class AbsClassifierInstance:

    def __init__(self, name):
        self.name = name
        self.ds_path = None
        self.test_percent = 0
        self.classifier = None
        self.model = None
        self.attrs = np.array([])
        self.classes = np.array([])
        self.train_x = np.array([])
        self.train_y = np.array([])
        self.test_x = np.array([])
        self.test_y = np.array([])
        self.plot_cls = False

    @property
    def train_size(self):
        return len(self.train_x)

    @property
    def test_size(self):
        return len(self.test_x)

    @property
    def n_features(self):
        return len(self.attrs) - 1

    @property
    def class_names(self):
        """class labels as text"""
        return [tu.text_label(cn) for cn in self.classes]

    @property
    def plot_filename(self):
        ds = self.ds_path or tu.DEFAULT_DS
        return f'{self.name}_{Path(ds).stem}.png'

    @staticmethod
    def formatter(x, y):
        return x

    def predict(self, data):
        return self.model.predict(data)

    def plot(self):
        pass

    def train(self, dataset, test_size):
        pass

    def train_stats(self):
        tu.show('Read dataset', self.ds_path)
        tu.show('Attributes', self.n_features)
        tu.show('Classifier', self.name)
        tu.show('Classes', ", ".join(self.class_names))
        tu.show('Training instances', self.train_size)
        tu.show('Test instances', self.test_size)

        if self.test_size > 0:
            predictions = self.predict(self.test_x)
            split = [str(np.count_nonzero(self.test_y == v))
                     for v in self.classes]
            tu.show('Test split', "/".join(split))
            tu.score(self.test_y, predictions, 0, display=True)

        if self.plot_cls:
            self.plot()
