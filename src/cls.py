import logging
from collections import namedtuple

from math import ceil
from os import path
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from src import BaseUtil

logger = logging.getLogger(__name__)


class AbsClassifierInstance(BaseUtil):

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
        self.mask_cols = []

    @property
    def n_train(self):
        return len(self.train_x)

    @property
    def n_test(self):
        return len(self.test_x)

    @property
    def n_features(self):
        return len(self.attrs) - 1

    @property
    def n_classes(self):
        return len(self.classes)

    @staticmethod
    def text_label(i):
        """convert text label to numeric"""
        return 'malicious' if i == 1 else 'benign'

    @staticmethod
    def int_label(text):
        """convert numeric label to text label"""
        return 0 if text.lower() == 'benign' else 1

    @property
    def class_names(self):
        """class labels as text"""
        return [self.text_label(cn) for cn in self.classes]

    @property
    def plot_path(self):
        fn = f'{self.name}_{Path(self.ds_path).stem}.png'
        return path.join(self.out_dir, fn)

    @property
    def mutable_attrs(self):
        return sorted([
            a for i, a in enumerate(self.attrs)
            if i not in self.mask_cols and i < self.n_features])

    @property
    def immutable_attrs(self):
        return sorted([
            a for i, a in enumerate(self.attrs)
            if i in self.mask_cols and i < self.n_features])

    @staticmethod
    def formatter(x, y):
        return x

    def determine_proto(self, record):
        """Determine protocol for some records, by scan of the attribute
        values, and active bit. Returns other if not found."""
        proto_label = next(
            (a for a, b in
             [(lbl, int(record[i])) for i, lbl
              in enumerate(self.attrs) if 'proto' in lbl]
             if b == 1), self.proto_other)
        if 'tcp' in proto_label:
            return self.tcp
        if 'udp' in proto_label:
            return self.udp
        return self.proto_other

    def set_mask_cols(self):
        indices = []
        for col_i in range(self.n_features):
            col_values = list(np.unique(self.train_x[:, col_i]))
            if set(col_values).issubset({0, 1}):
                indices.append(col_i)
        self.mask_cols = indices

    def predict(self, data):
        return self.model.predict(data)

    def prep_model(self, robust):
        pass

    def prep_classifier(self):
        pass

    def tree_plotter(self):
        pass

    def plot(self):
        self.ensure_out_dir(self.out_dir)
        self.tree_plotter()
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=200)

    def load(self, dataset_path, test_percent=0):
        self.ds_path = dataset_path
        self.test_percent = test_percent
        self.__load_csv_data()
        self.set_mask_cols()
        return self

    def train(self, robust=False):
        self.prep_model(robust)
        self.prep_classifier()

        self.show('Read dataset', self.ds_path)
        self.show('Attributes', self.n_features)
        self.show('Classifier', self.name)
        self.show('Classes', ", ".join(self.class_names))
        self.show('Training size', self.n_train)
        self.show('Test size', self.n_test)

        # evaluate performance
        records = (
            (self.test_x, self.test_y)
            if self.n_test > 0 else
            (self.train_x, self.train_y))
        predictions = self.predict(self.formatter(*records))
        self.score(records[1], predictions, display=True)

        return self

    def normalize(self, data, capture=False):
        """normalize values in range 0.0 - 1.0."""
        np.seterr(divide='ignore', invalid='ignore')
        for i in range(self.n_features):
            range_max = max(data[:, i])
            if capture:
                self.attr_ranges[i] = ceil(range_max)
            data[:, i] = (data[:, i]) / range_max
            data[:, i] = np.nan_to_num(data[:, i])
        return data

    def denormalize(self, data):
        """Denormalize values to original value range."""
        data_copy = data.copy()
        for i in range(self.n_features):
            range_max = self.attr_ranges[i]
            data_copy[:, i] = range_max * (data[:, i])
        return data_copy

    @staticmethod
    def attr_fix(attrs):
        return [a.replace(' ', '').replace('=', '_').replace('-', '')
                for a in attrs]

    def __load_csv_data(self, max_size=-1):
        """
        Read dataset and split to train/test using random sampling.
        """

        df = pd.read_csv(self.ds_path).fillna(0)
        self.attrs = AbsClassifierInstance.attr_fix(
            [col for col in df.columns])
        self.test_x, self.test_y = np.array([]), np.array([])
        split = 0 < self.test_percent < len(df)

        # sample training/test instances
        if split:
            train, test = train_test_split(
                df, test_size=self.test_percent)
            self.test_x = self.normalize(np.array(test)[:, :-1])
            self.test_y = np.array(test)[:, -1].astype(int).flatten()
        else:
            train = df

        train_x = self.normalize(np.array(train)[:, :-1], capture=True)
        train_y = np.array(train)[:, -1].astype(int).flatten()

        if max_size > 0:
            train_x = train_x[: max_size, :]
            train_y = train_y[: max_size]

        self.classes = np.unique(train_y)
        self.train_x = train_x
        self.train_y = train_y

    @staticmethod
    def score(true_labels, predictions, positive=0, display=False):
        """Calculate performance metrics."""
        sc, tp_tn, num_pos_pred, num_pos_actual = 0, 0, 0, 0
        for actual, pred in zip(true_labels, predictions):
            int_pred = int(round(pred, 0))
            if int_pred == positive:
                num_pos_pred += 1
            if actual == positive:
                num_pos_actual += 1
            if int_pred == actual:
                tp_tn += 1
            if int_pred == actual and int_pred == positive:
                sc += 1

        accuracy = tp_tn / len(predictions)
        precision = 1 if num_pos_pred == 0 else sc / num_pos_pred
        recall = 1 if num_pos_actual == 0 else sc / num_pos_actual
        f_score = (2 * precision * recall) / (precision + recall)

        if display:
            AbsClassifierInstance.show('Accuracy', f'{accuracy * 100:.2f} %')
            AbsClassifierInstance.show('Precision', f'{precision * 100:.2f} %')
            AbsClassifierInstance.show('Recall', f'{recall * 100:.2f} %')
            AbsClassifierInstance.show('F-score', f'{f_score * 100:.2f} %')
