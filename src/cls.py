import logging
from math import ceil
from os import path
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import utility as tu

logger = logging.getLogger(__name__)


class AbsClassifierInstance:

    def __init__(self, name, out):
        self.name = name
        self.model = None
        self.classifier = None
        self.ds_path = None
        self.test_percent = 0
        self.attrs = np.array([])
        self.classes = np.array([])
        self.train_x = np.array([])
        self.train_y = np.array([])
        self.test_x = np.array([])
        self.test_y = np.array([])
        self.attr_ranges = {}
        self.out_dir = out

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
    def n_classes(self):
        return len(self.classes)

    @property
    def class_names(self):
        """class labels as text"""
        return [tu.text_label(cn) for cn in self.classes]

    @property
    def plot_path(self):
        fn = f'{self.name}_{Path(self.ds_path).stem}.png'
        return path.join(self.out_dir, fn)

    @staticmethod
    def formatter(x, y):
        return x

    def predict(self, data):
        return self.model.predict(data)

    def prep_model(self, robust):
        pass

    def prep_classifier(self):
        pass

    def tree_plotter(self):
        pass

    def plot(self):
        tu.ensure_out_dir(self.out_dir)
        self.tree_plotter()
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=200)

    def load(self, dataset_path, test_percent=0):
        self.ds_path = dataset_path
        self.test_percent = test_percent
        self.__load_csv_data()
        return self

    def train(self, robust=False):
        self.prep_model(robust)
        self.prep_classifier()

        tu.show('Read dataset', self.ds_path)
        tu.show('Attributes', self.n_features)
        tu.show('Classifier', self.name)
        tu.show('Classes', ", ".join(self.class_names))
        tu.show('Training instances', self.train_size)
        tu.show('Test instances', self.test_size)

        # evaluate performance
        records = (
            (self.test_x, self.test_y)
            if self.test_size > 0 else
            (self.train_x, self.train_y))
        predictions = self.predict(self.formatter(*records))
        tu.score(records[1], predictions, display=True)

        return self

    def normalize(self, data, capture=False):
        """normalize values in range 0.0 - 1.0."""
        np.seterr(divide='ignore', invalid='ignore')
        for i in range(len(data[0])):
            range_max = max(data[:, i])
            if capture:
                self.attr_ranges[i] = ceil(range_max)
            data[:, i] = (data[:, i]) / range_max
            data[:, i] = np.nan_to_num(data[:, i])
        return data

    def __load_csv_data(self, max_size=-1):
        """
        Read dataset and split to train/test using random sampling.
        """

        df = pd.read_csv(self.ds_path).fillna(0)
        attrs = [col for col in df.columns]
        split = 0 < self.test_percent < len(df)
        test_x, test_y = np.array([]), np.array([])

        # sample training/test instances
        if split:
            train, test = train_test_split(
                df, test_size=self.test_percent)
            test_x = self.normalize(np.array(test)[:, :-1])
            test_y = np.array(test)[:, -1].astype(int).flatten()
        else:
            train = df

        train_x = self.normalize(np.array(train)[:, :-1], capture=True)
        train_y = np.array(train)[:, -1].astype(int).flatten()

        if max_size > 0:
            train_x = train_x[: max_size, :]
            train_y = train_y[: max_size]

        self.attrs = attrs
        self.classes = np.unique(train_y)
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
