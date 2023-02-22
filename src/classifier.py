import warnings

warnings.filterwarnings("ignore")

from typing import Union, Optional

import numpy as np
from art.estimators.classification import BlackBoxClassifier, \
    CatBoostARTClassifier, DetectorClassifier, EnsembleClassifier, \
    PyTorchClassifier, TensorFlowClassifier, TensorFlowV2Classifier, \
    GPyGaussianProcessClassifier, LightGBMClassifier, XGBoostClassifier, \
    KerasClassifier, MXClassifier
from art.estimators.classification.classifier import ClassifierNeuralNetwork
from art.estimators.classification.scikitlearn import \
    ScikitlearnDecisionTreeClassifier, ScikitlearnAdaBoostClassifier, \
    ScikitlearnExtraTreesClassifier, ScikitlearnGradientBoostingClassifier, \
    ScikitlearnLogisticRegression, ScikitlearnClassifier, \
    ScikitlearnExtraTreeClassifier, ScikitlearnBaggingClassifier, \
    ScikitlearnRandomForestClassifier, ScikitlearnSVC
from art.experimental.estimators.classification import JaxClassifier

CLS_TYPE = Optional[Union[
    BlackBoxClassifier, CatBoostARTClassifier, DetectorClassifier,
    EnsembleClassifier, GPyGaussianProcessClassifier, KerasClassifier,
    JaxClassifier, LightGBMClassifier, MXClassifier, PyTorchClassifier,
    ScikitlearnClassifier, ScikitlearnDecisionTreeClassifier,
    ScikitlearnExtraTreeClassifier, ScikitlearnAdaBoostClassifier,
    ScikitlearnBaggingClassifier, ScikitlearnExtraTreesClassifier,
    ScikitlearnGradientBoostingClassifier, ScikitlearnRandomForestClassifier,
    ScikitlearnLogisticRegression, ScikitlearnSVC, TensorFlowClassifier,
    TensorFlowV2Classifier, XGBoostClassifier, ClassifierNeuralNetwork]]


class Classifier:

    def __init__(
            self, name, out, attrs, y, robust, mask_cols, attr_ranges
    ):
        self.name = name
        self.out_dir = out
        self.attrs = attrs[:]
        self.classes = np.unique(y)
        self.classifier: CLS_TYPE = None
        self.model = None
        self.train_x = np.array([])
        self.train_y = np.array([])
        self.test_x = np.array([])
        self.test_y = np.array([])
        self.attr_ranges = attr_ranges
        self.mask_cols = mask_cols
        self.fold_n = 1
        self.robust = robust
        self.n_pred = 0
        self.n_true_pn = 0
        self.n_actl_pos = 0
        self.n_pred_pos = 0
        self.n_true_p = 0

    def reset(self):
        self.classifier = None
        self.model = None
        self.train_x = np.array([])
        self.train_y = np.array([])
        self.test_x = np.array([])
        self.test_y = np.array([])
        self.fold_n = 1
        self.n_pred = 0
        self.n_true_pn = 0
        self.n_actl_pos = 0
        self.n_pred_pos = 0
        self.n_true_p = 0
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
    def mutable_attrs(self):
        return sorted([
            a for i, a in enumerate(self.attrs)
            if i not in self.mask_cols and i < self.n_features])

    @property
    def immutable_attrs(self):
        return sorted([
            a for i, a in enumerate(self.attrs)
            if i in self.mask_cols and i < self.n_features])

    @property
    def accuracy(self):
        return -1 if self.n_pred == 0 else \
            self.n_true_pn / self.n_pred

    @property
    def precision(self):
        return 1 if self.n_pred_pos == 0 else \
            self.n_true_p / self.n_pred_pos

    @property
    def recall(self):
        return 1 if self.n_actl_pos == 0 else \
            self.n_true_p / self.n_actl_pos

    @property
    def f_score(self):
        pr = self.precision + self.recall
        return 0 if pr == 0 else \
            (2 * self.precision * self.recall) / pr

    @staticmethod
    def formatter(x, y):
        """Format records to fit this classifier, if necessary.

           The default format is np.array. If the implementing classifier needs
           a different data format (e.g., DMatrix), then apply it here.
        """
        return x

    def predict(self, record):
        """Predict label for specified record."""
        return self.model.predict(record)

    def init_learner(self, robust):
        """Implement model training phase. If robust is true, the model should
        apply appropriate defense. This is called after data has been loaded
        and is ready; cf. `Classifier.train`.
        """
        pass

    def load(self, x, y, fold_train, fold_test, fold_n):
        self.train_x = self.normalize(x[fold_train, :])
        self.train_y = y[fold_train].astype(int).flatten()
        self.test_x = self.normalize(x[fold_test, :])
        self.test_y = y[fold_test].astype(int).flatten()
        self.classes = np.unique(y)
        self.fold_n = fold_n
        return self

    def train(self):
        self.init_learner(self.robust)
        records = ((self.test_x, self.test_y) if self.n_test > 0
                   else (self.train_x, self.train_y))
        predictions = self.predict(self.formatter(*records))
        self.score(records[1], predictions)
        return self

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

    def score(self, true_labels, predictions, positive=0):
        """Calculate performance metrics."""
        tp, tp_tn, p_pred, p_actual = 0, 0, 0, 0
        for actual, pred in zip(true_labels, predictions):
            int_pred = int(round(pred, 0))
            if int_pred == positive:
                p_pred += 1
            if actual == positive:
                p_actual += 1
            if int_pred == actual:
                tp_tn += 1
            if int_pred == actual and int_pred == positive:
                tp += 1
        self.n_pred = len(predictions)
        self.n_actl_pos = p_actual
        self.n_pred_pos = p_pred
        self.n_true_pn = tp_tn
        self.n_true_p = tp
