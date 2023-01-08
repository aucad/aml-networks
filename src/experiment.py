import json
import logging
import time
from collections import namedtuple
from random import sample
from typing import Tuple, List

import numpy as np
from sklearn.model_selection import KFold

# noinspection PyUnresolvedReferences
from src import Classifier, Attack, Validator, utility, \
    HopSkip, Zoo, DecisionTree, XgBoost, Show, Ratio, sdiv, machine

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
                return XgBoost(*args)

    class AttackLoader:
        """Load selected attack mode."""
        HOP_SKIP = 'hsj'
        ZOO = 'zoo'

        @staticmethod
        def load(kind, *args) -> Attack:
            if kind == Experiment.AttackLoader.HOP_SKIP:
                return HopSkip(*args)
            else:
                return Zoo(*args)

    class Result(object):
        """Track average results of K-folds."""

        def __init__(self):
            self.__accuracy = []
            self.__precision = []
            self.__recall = []
            self.__f_score = []
            self.__n_records = []
            self.__n_evasions = []
            self.__n_valid = []
            self.__labels = []
            self.__validations = []
            self.__errors = []
            self.__proto_init = None
            self.__proto_evasions = []
            self.__proto_valid = []

        def append_attack(self, attack: Attack):
            self.__n_evasions.append(attack.n_evasions)
            self.__n_valid.append(attack.n_valid)
            self.__n_records.append(attack.n_records)
            self.__labels.append(attack.label_stats)
            self.__validations.append(attack.validation_reasons)
            self.__errors.append(attack.error)
            self.__proto_evasions.append(attack.adv_proto)
            self.__proto_valid.append(attack.adv_proto_valid)
            if not self.__proto_init:
                self.__proto_init = attack.init_proto

        def append_cls(self, cls: Classifier):
            self.__accuracy.append(cls.accuracy)
            self.__precision.append(cls.precision)
            self.__recall.append(cls.recall)
            self.__f_score.append(cls.f_score)

        @staticmethod
        def calc_average(arr) -> float:
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
        self.mask_cols = []
        self.attr_ranges = {}
        self.stats = Experiment.Result()
        config_keys = ",".join(kwargs.keys())
        self.config = namedtuple('exp', config_keys)(**kwargs)

    @property
    def n_records(self) -> int:
        return len(self.X)

    @property
    def n_attr(self) -> int:
        return len(self.attrs)

    @property
    def duration(self) -> Tuple[int, float]:
        seconds = round((self.end_time - self.start_time) / 1e9, 6)
        minutes = int(seconds // 60)
        seconds = seconds - (minutes * 60)
        return minutes, seconds

    @property
    def output_file(self):
        return utility.generate_name(self.uuid, self.config, 'json')

    def load_csv(self, ds_path: str, n_splits: int):
        self.attrs, rows = utility.read_dataset(ds_path)
        self.X = rows[:, :-1]
        self.y = rows[:, -1].astype(int).flatten()
        self.folds = [(tr_i, ts_i) for tr_i, ts_i
                      in KFold(n_splits=n_splits).split(self.X)]
        self.mask_cols, n_feat = [], len(self.attrs) - 1
        for col_i in range(n_feat):
            self.attr_ranges[col_i] = max(self.X[:, col_i])
            col_values = list(np.unique(self.X[:, col_i]))
            if set(col_values).issubset({0, 1}):
                self.mask_cols.append(col_i)

    def do_fold(self, fold_num: int, fold_indices: List[int]):

        def attack_round(sample_size):
            sample_idx = None if sample_size < 1 else \
                sample(range(0, self.cls.n_test), sample_size)
            self.attack.reset().set_cls(self.cls, sample_idx).run()
            self.stats.append_attack(self.attack)

        self.cls.reset().load(
            self.X.copy(), self.y.copy(),
            *fold_indices, fold_num).train()
        self.stats.append_cls(self.cls)
        self.log_fold_result(fold_num)
        ss = 0 if self.config.sample_size < 1 else \
            min(self.config.sample_size, self.cls.n_test)

        if self.attack:
            for n in range(self.config.sample_times):
                attack_round(ss)
                self.log_fold_attack(n + 1)

    def run(self):
        config = self.config
        self.load_csv(config.dataset, config.folds)
        cls_args = (config.cls, config.out, self.attrs, self.y,
                    config.robust, self.mask_cols, self.attr_ranges)
        atk_args = (config.attack, config.validator, self.uuid,
                    config.capture, config.iter, config.silent)
        self.cls = Experiment.ClsLoader.init(*cls_args)
        self.attack = Experiment.AttackLoader.load(*atk_args) \
            if config.attack else None
        self.log_experiment_setup()

        if config.validator:
            Validator.validate_dataset(
                config.validator, config.dataset)

        self.start_time = time.time_ns()
        for i, fold in enumerate(self.folds):
            self.do_fold(i + 1, fold)
        self.end_time = time.time_ns()
        self.log_experiment_result()
        self.save_result()

    def log_experiment_setup(self):
        Show('Dataset', self.config.dataset)
        Show('Record count', self.n_records)
        Show('Attributes', self.n_attr)
        Show('Classifier', self.cls.name)
        Show('Robust', self.config.robust)
        Show('Classes', ", ".join(self.cls.class_names))
        Show('K-folds', self.config.folds)
        Show('Mutable', ", ".join(self.cls.mutable_attrs))
        Show('Immutable', ", ".join(self.cls.immutable_attrs))
        if self.attack:
            Show('Attack', self.attack.name)
            Show('Attack max iter', self.attack.max_iter)
            if self.config.sample_size > 0:
                Show('Records sample',
                     f'{self.config.sample_size} x '
                     f'{self.config.sample_times}')

    def log_fold_result(self, fold_n: int):
        print('=' * 52)
        Show('Fold', fold_n)
        Show('Accuracy', f'{self.cls.accuracy * 100:.2f} %')
        Show('Precision', f'{self.cls.precision * 100:.2f} %')
        Show('Recall', f'{self.cls.recall * 100:.2f} %')
        Show('F-score', f'{self.cls.f_score * 100:.2f} %')

    def log_fold_attack(self, sample_n: int):
        Show('Sample', f'{sample_n} of {self.config.sample_times}')
        if self.attack:
            Ratio('Evasions', self.attack.n_evasions,
                  self.attack.n_records)
            if self.attack.use_validator:
                Ratio('Valid', self.attack.n_valid,
                      self.attack.n_evasions)
            if self.attack.has_evasions:
                Show('Class labels',
                     utility.dump_num_dict(self.attack.label_stats))
            if self.attack.has_invalid:
                Show('Invalid reasons',
                     utility.dump_num_dict(
                         self.attack.validation_reasons))

    def log_experiment_result(self):
        print('=' * 52, '')
        Show('Avg. Accuracy', f'{(self.stats.accuracy * 100):.2f} %')
        Show('Avg. Precision', f'{(self.stats.precision * 100):.2f} %')
        Show('Avg. Recall', f'{(self.stats.recall * 100):.2f} %')
        Show('Avg. F-score', f'{(self.stats.f_score * 100):.2f} %')
        if self.attack:
            Ratio('Evasions', self.stats.n_evasions,
                  self.stats.n_records)
            if self.attack.use_validator:
                Ratio('Valid', self.stats.n_valid,
                      self.stats.n_evasions)
        Show('Time', "{0} min {1:.2f} s".format(*self.duration))

    def to_dict(self) -> dict:
        """Dictionary representation of an experiment."""
        attack_attrs = {} if not self.attack else \
            {'max_iter': self.attack.max_iter}
        return {
            'dataset': self.config.dataset,
            'dataset_name': utility.name_only(self.config.dataset),
            'n_records': self.n_records,
            'n_attributes': self.n_attr,
            'attrs': self.attrs,
            'immutable': self.mask_cols,
            'attr_mutable': self.cls.mutable_attrs,
            'attr_immutable': self.cls.immutable_attrs,
            'robust': self.config.robust,
            'classes': self.cls.class_names,
            'k_folds': self.config.folds,
            'validator': self.config.validator,
            'classifier': self.config.cls,
            'sample_size': self.config.sample_size,
            'sample_times': self.config.sample_times,
            'attack': self.config.attack,
            'attr_ranges': dict(
                zip(self.attrs, self.attr_ranges.values())),
            'start': self.start_time,
            'end': self.end_time,
            'current_utc': int(time.time_ns()),
            **attack_attrs,
            **self.stats.__dict__,
            'machine': machine.machine_details()
        }

    def save_result(self):
        if not self.config.no_save:
            with open(self.output_file, "w") as outfile:
                json.dump(self.to_dict(), outfile, indent=4)
            if not self.config.silent:
                print('Saved result to', self.output_file)
