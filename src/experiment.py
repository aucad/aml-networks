import logging
from collections import namedtuple
from typing import Tuple

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import KFold

# noinspection PyUnresolvedReferences
from src import Classifier, Attack, Validator, \
    HopSkip, Zoo, DecisionTree, XGB, Show, Ratio, sdiv, utility

logger = logging.getLogger(__name__)


class Experiment:
    class ClsLoader:
        """Load selected classifier."""
        XGBOOST = 'xgb'
        DECISION_TREE = 'tree'

        @staticmethod
        def init(kind, *args) -> Classifier:
            if kind == Experiment.ClsLoader.DECISION_TREE:
                return DecisionTree(*args)
            else:
                return XGB(*args)

    class AttackLoader:
        """Load selected attack mode."""
        HOP_SKIP = 'hop'
        ZOO = 'zoo'

        @staticmethod
        def load(kind, *args) -> Attack:
            if kind == Experiment.AttackLoader.HOP_SKIP:
                return HopSkip(*args)
            else:
                return Zoo(*args)

    class Stats:
        """Track average results of K-folds."""

        def __init__(self):
            self.__accuracy = []
            self.__precision = []
            self.__recall = []
            self.__f_score = []
            self.__n_records = []
            self.__n_evasions = []
            self.__n_valid = []

        def append_attack(self, attack: Attack):
            self.__n_evasions.append(attack.n_evasions)
            self.__n_valid.append(attack.n_valid)
            self.__n_records.append(attack.n_records)

        def append_cls(self, cls: Classifier):
            self.__accuracy.append(cls.accuracy)
            self.__precision.append(cls.precision)
            self.__recall.append(cls.recall)
            self.__f_score.append(cls.f_score)

        @staticmethod
        def calc_average(arr):
            return sdiv(sum(arr), len(arr))

        @property
        def accuracy(self) -> float:
            return self.calc_average(self.__accuracy)

        @property
        def precision(self) -> float:
            return self.calc_average(self.__precision)

        @property
        def recall(self) -> float:
            return self.calc_average(self.__recall)

        @property
        def f_score(self) -> float:
            return self.calc_average(self.__f_score)

        @property
        def n_evasions(self) -> float:
            return self.calc_average(self.__n_evasions)

        @property
        def n_valid(self) -> float:
            return self.calc_average(self.__n_valid)

        @property
        def n_records(self) -> float:
            return self.calc_average(self.__n_records)

        @property
        def evasion_ratio(self) -> float:
            return sdiv(self.n_evasions, self.n_records)

        @property
        def valid_ratio(self) -> float:
            return sdiv(self.n_valid, self.n_evasions)

    DEFAULT_DS = 'data/CTU-1-1.csv'
    DEFAULT_CLS = ClsLoader.XGBOOST
    ATTACKS = [AttackLoader.HOP_SKIP, AttackLoader.ZOO]
    CLASSIFIERS = [ClsLoader.DECISION_TREE, ClsLoader.XGBOOST]
    VALIDATORS = [Validator.NB15, Validator.IOT23]

    def __init__(self, uuid, **kwargs):
        self.uuid = uuid
        self.start_time = 0
        self.end_time = 0
        self.cls = None
        self.attack = None
        self.attrs = []
        self.X = None
        self.y = None
        self.folds = None
        self.stats = Experiment.Stats()
        config_keys = ",".join(kwargs.keys())
        self.config = namedtuple('exp', config_keys)(**kwargs)

    @property
    def n_records(self):
        return len(self.X)

    @property
    def n_attr(self):
        return len(self.attrs)

    @property
    def duration(self) -> Tuple[int, float]:
        seconds = round((self.end_time - self.start_time) / 1e9, 6)
        minutes = int(seconds // 60)
        seconds = seconds - (minutes * 60)
        return minutes, seconds

    def load_csv(self, ds_path, n_splits):
        df = pd.read_csv(ds_path).fillna(0)
        self.attrs = [col for col in df.columns]
        self.X = (np.array(df)[:, :-1])
        self.y = np.array(df)[:, -1].astype(int).flatten()
        self.folds = [(tr_i, ts_i) for tr_i, ts_i
                      in KFold(n_splits=n_splits).split(self.X)]

    def do_fold(self, fold_num, fold_indices):
        self.cls.reset().load(
            self.X.copy(), self.y.copy(),
            *fold_indices, fold_num).train()
        self.stats.append_cls(self.cls)
        if self.attack:
            self.attack.reset().set_cls(self.cls).run()
            self.stats.append_attack(self.attack)
        self.log_fold_result(fold_num)

    def run(self):
        config = self.config
        self.load_csv(config.dataset, config.folds)
        cls_args = (config.cls, config.out, self.attrs, self.X,
                    self.y, config.dataset, config.robust)
        atk_args = (config.attack, False, False,
                    config.validator, self.uuid,
                    config.capture, config.iter)

        self.cls = Experiment.ClsLoader.init(*cls_args)
        self.attack = Experiment.AttackLoader.load(*atk_args) \
            if config.attack else None
        self.log_experiment_setup()

        self.start_time = time.time_ns()
        for i, fold in enumerate(self.folds):
            self.do_fold(i + 1, fold)
        self.end_time = time.time_ns()

        self.log_experiment_result()

    def log_experiment_setup(self):
        Show('Dataset', self.config.dataset)
        Show('Record count', self.n_records)
        Show('Attributes', self.n_attr)
        Show('Classifier', self.cls.name)
        Show('Robust', self.config.robust)
        Show('Classes', ", ".join(self.cls.class_names))
        Show('Attack', self.attack.name)
        Show('K-folds', self.config.folds)
        Show('Attack max iter', self.attack.max_iter)
        Show('Mutable', ", ".join(self.cls.mutable_attrs))
        Show('Immutable', ", ".join(self.cls.immutable_attrs))
        Show('=' * 52, '')

    def log_fold_result(self, fold_n):
        Show('Fold', fold_n)
        Show('Accuracy', f'{self.cls.accuracy * 100:.2f} %')
        Show('Precision', f'{self.cls.precision * 100:.2f} %')
        Show('Recall', f'{self.cls.recall * 100:.2f} %')
        Show('F-score', f'{self.cls.f_score * 100:.2f} %')
        Ratio('Evasions', self.attack.n_evasions, self.attack.n_records)
        if self.attack.use_validator:
            Ratio('Valid', self.attack.n_valid, self.attack.n_evasions)
        if self.attack.n_evasions > 0:
            ls = self.attack.printable_label_stats()
            Show('Class labels', '\n'.join(ls))
        if self.attack.n_evasions != self.attack.n_valid:
            Validator.dump_reasons(self.attack.validation_reasons)
        Show('L-norm', "{0:.6f} - {1:.6f}".format(*self.attack.error))

    def log_experiment_result(self):
        Show('=' * 52, '')
        Show('Avg. Accuracy', f'{(self.stats.accuracy * 100):.2f} %')
        Show('Avg. Precision', f'{(self.stats.precision * 100):.2f} %')
        Show('Avg. Recall', f'{(self.stats.recall * 100):.2f} %')
        Show('Avg. F-score', f'{(self.stats.f_score * 100):.2f} %')
        Ratio('Evasions', self.stats.n_evasions, self.stats.n_records)
        if self.attack.use_validator:
            Ratio('Valid', self.stats.n_valid, self.stats.n_evasions)
        Show('Time', "{0} min {1:.2f} s".format(*self.duration))
