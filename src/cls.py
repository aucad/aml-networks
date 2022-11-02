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

    def __init__(self, name, out, attrs, x, y, ds_path, robust=False):
        self.name = name
        self.out_dir = out
        self.attrs = self.attr_fix(attrs[:])
        self.classes = np.unique(y)
        self.ds_path = ds_path
        self.classifier = None
        self.model = None
        self.classes = np.unique(y)
        self.train_x = np.array([])
        self.train_y = np.array([])
        self.test_x = np.array([])
        self.test_y = np.array([])
        self.attr_ranges = {}
        self.mask_cols = []
        self.fold_n = 1
        self.capture_ranges(x)
        self.set_mask_cols(x)
        self.robust = robust
        self.show('Classifier', self.name)
        self.show('Robust', self.robust)
        self.show('Classes', ", ".join(self.class_names))
        self.show('Mutable', ", ".join(self.mutable_attrs))
        self.show('Immutable', ", ".join(self.immutable_attrs))
        self.stats = []

    def reset(self):
        self.classifier = None
        self.model = None
        self.train_x = np.array([])
        self.train_y = np.array([])
        self.test_x = np.array([])
        self.test_y = np.array([])
        self.fold_n = 1
        self.stats = []
        return self

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

    def set_mask_cols(self, X):
        indices = []
        for col_i in range(self.n_features):
            col_values = list(np.unique(X[:, col_i]))
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

    def load(self, X, y, fold_train, fold_test, fold_n):
        self.train_x = self.normalize(X[fold_train, :])
        self.train_y = y[fold_train].astype(int).flatten()
        self.test_x = self.normalize(X[fold_test, :])
        self.test_y = y[fold_test].astype(int).flatten()
        self.classes = np.unique(y)
        self.fold_n = fold_n
        return self

    def train(self):
        self.prep_model(self.robust)
        self.prep_classifier()
        self.show('K-fold', self.fold_n)

        # evaluate performance
        records = (
            (self.test_x, self.test_y)
            if self.n_test > 0 else
            (self.train_x, self.train_y))
        predictions = self.predict(self.formatter(*records))
        self.stats = self.score(records[1], predictions, display=True)

        return self

    def capture_ranges(self, data):
        """normalize values in range 0.0 - 1.0."""
        np.seterr(divide='ignore', invalid='ignore')
        for i in range(self.n_features):
            range_max = max(data[:, i])
            self.attr_ranges[i] = ceil(range_max)

    def normalize(self, data):
        """normalize values in range 0.0 - 1.0."""
        np.seterr(divide='ignore', invalid='ignore')
        for i in range(self.n_features):
            range_max = self.attr_ranges[i]
            data[:, i] = (data[:, i]) / range_max
            data[:, i] = np.nan_to_num(data[:, i])
        return data

    def denormalize(self, data):
        """Denormalize values to original value range."""
        data_copy = data.copy()
        for i in range(self.n_features):
            range_max = self.attr_ranges[i]
            data_copy[:, i] = np.rint(range_max * (data[:, i]))
        return data_copy

    @staticmethod
    def attr_fix(attrs):
        """Remove selected special chars from attributes so that
        the remaining forms a valid Python identifier."""
        return [a.replace(' ', '')
                    .replace('=', '_')
                    .replace('-', '')
                    .replace('^', '_')
                    .replace('conn_state_other', 'conn_state_OTH')
                for a in attrs]

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
            BaseUtil.show('Accuracy', f'{accuracy * 100:.2f} %')
            BaseUtil.show('Precision', f'{precision * 100:.2f} %')
            BaseUtil.show('Recall', f'{recall * 100:.2f} %')
            BaseUtil.show('F-score', f'{f_score * 100:.2f} %')
        return [accuracy, precision, recall, f_score]
