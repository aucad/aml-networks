import json
import gc
import logging
import os
import time
from collections import namedtuple
from random import sample
from typing import Tuple, List

import numpy as np
from sklearn.model_selection import KFold

# noinspection PyUnresolvedReferences
from aml import Classifier, Attack, Validator, utility, \
    HopSkip, Zoo, NeuralNetwork, XgBoost, Show, Ratio, sdiv, machine

logger = logging.getLogger(__name__)


class ClsLoader:
    """Load selected classifier."""
    XGB = 'xgb'
    DNN = 'dnn'

    @staticmethod
    def init(kind, *args) -> Classifier:
        if kind == ClsLoader.DNN:
            return NeuralNetwork(*args)
        else:
            return XgBoost(*args)


class AttackLoader:
    """Load selected attack mode."""
    HSJ = 'hsj'
    ZOO = 'zoo'

    @staticmethod
    def load(kind, *args) -> Attack:
        if kind == AttackLoader.HSJ:
            return HopSkip(*args)
        else:
            return Zoo(*args)


class Experiment:
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
            self.__proto_init = []
            self.__proto_evasions = []
            self.__proto_valid = []

        def append_attack(self, attack: Attack):
            self.__n_evasions.append(attack.n_evasions)
            self.__n_valid.append(attack.n_valid)
            self.__n_records.append(attack.n_records)
            self.__labels.append(attack.label_stats)
            self.__validations.append(attack.validation_reasons)
            self.__errors.append(attack.error)
            self.__proto_init.append(attack.init_proto)
            self.__proto_evasions.append(attack.adv_proto)
            self.__proto_valid.append(attack.adv_proto_valid)

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

    DEFAULT_DS = 'data/CTU.csv'
    DEFAULT_CLS = ClsLoader.XGB
    DEFAULT_CF = 'config/default.yaml'
    ATTACKS = [AttackLoader.HSJ, AttackLoader.ZOO]
    CLASSIFIERS = [ClsLoader.XGB, ClsLoader.DNN]
    VALIDATORS = [Validator.NB15, Validator.IOT23]

    def __init__(self, uuid, config_obj, **kwargs):
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
        config_keys = ",".join(
            list(kwargs.keys()) + list(config_obj.keys()))
        self.config = (namedtuple('exp', config_keys)
                       (**kwargs, **config_obj))

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

    @property
    def is_repeat(self):
        gen_name = utility.generate_name('', self.config, None)
        match = gen_name.split('/').pop()
        prev = [f for f in os.listdir(self.config.out)
                if f.startswith(match)]
        return prev[0] if len(prev) > 0 else None

    def custom_config(self, key):
        return getattr(self.config, key) \
            if key and hasattr(self.config, key) else None

    def load_csv(self, ds_path: str, n_splits: int):
        self.attrs, rows = utility.read_dataset(ds_path)
        self.X = rows[:, :-1]
        self.y = rows[:, -1].astype(int).flatten()
        self.mask_cols, n_feat = [], len(self.attrs) - 1
        self.folds = [x for x in KFold(
            n_splits=n_splits, shuffle=True).split(self.X)]
        for col_i in range(n_feat):
            self.attr_ranges[col_i] = max(self.X[:, col_i])
            col_values = list(np.unique(self.X[:, col_i]))
            if set(col_values).issubset({0, 1}):
                self.mask_cols.append(col_i)

    def run(self):
        config, prev = self.config, self.is_repeat
        if config.resume and prev:
            return print('Saved result to', prev)
        self.load_csv(config.dataset, config.folds)
        cls_args = (config.cls, config.out, self.attrs, self.y,
                    config.robust, self.mask_cols, self.attr_ranges,
                    self.custom_config(config.cls))
        atk_args = (config.attack, config.validator, self.uuid,
                    config.capture, config.iter, config.silent,
                    self.custom_config(config.attack))
        self.cls = ClsLoader.init(*cls_args)
        self.attack = AttackLoader.load(*atk_args) \
            if config.attack else None
        self.log_experiment_setup()

        if config.validator:
            Validator.validate_dataset(
                config.validator, config.dataset)

        self.start_time = time.time_ns()
        for i, fold in enumerate(self.folds):
            self.exec_fold(i + 1, fold)
            self.cls.cleanup()
            gc.collect()
        self.end_time = time.time_ns()
        self.log_experiment_result()
        self.save_result()
        self.cleanup()

    def exec_fold(self, fold_num: int, fold_indices: List[int]):
        self.cls.reset() \
            .load(self.X.copy(), self.y.copy(), *fold_indices, fold_num) \
            .train()
        self.stats.append_cls(self.cls)
        self.log_training_result(fold_num)
        sample_size = 0 if self.config.sample_size < 1 else \
            min(self.config.sample_size, self.cls.n_test)

        if self.attack:
            for n in range(self.config.sample_times):
                sample_idx = None if sample_size < 1 else \
                    sample(range(0, self.cls.n_test), sample_size)
                self.attack.reset().set_cls(self.cls, sample_idx).run()
                self.attack.eval_examples()
                self.attack.validate()
                if self.attack.save_records and self.attack.n_evasions > 0:
                    self.attack.dump_result(self.config.out)
                self.stats.append_attack(self.attack)
                gc.collect()
                self.log_fold_attack(n + 1, self.config.sample_times)

    def log_experiment_setup(self):
        Show('Dataset', self.config.dataset)
        Show('Record count', self.n_records)
        Show('Attributes', self.n_attr)
        Show('Classifier', self.cls.name)
        Show('Robust', self.config.robust)
        Show('Classes', ", ".join(self.cls.class_names))
        Show('K-folds', self.config.folds)
        Show('Mutable', len(self.cls.mutable))
        Show('Immutable', len(self.cls.immutable))
        if self.attack:
            Show('Attack', self.attack.name)
            Show('Attack max iter', self.attack.max_iter)
            if self.config.sample_size > 0:
                Show('Records sample',
                     f'{self.config.sample_size} x '
                     f'{self.config.sample_times}')

    def log_training_result(self, fold_n: int):
        print('=' * 52)
        Show('Fold', fold_n)
        Show('Accuracy', f'{self.cls.accuracy * 100:.2f} %')
        Show('Precision', f'{self.cls.precision * 100:.2f} %')
        Show('Recall', f'{self.cls.recall * 100:.2f} %')
        Show('F-score', f'{self.cls.f_score * 100:.2f} %')

    def log_fold_attack(self, sample_n: int, n_total: int):
        if n_total > 1:
            print('-' * 5)
        if n_total > 1:
            Show('Sampling round',
                 f'{sample_n}/{self.config.sample_times}')
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
                Show('Invalid reasons', utility.dump_num_dict(
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
            'attr_mutable': self.cls.mutable,
            'attr_immutable': self.cls.immutable,
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
                print('Saved result to', self.output_file, '\n')

    def cleanup(self):
        del self.cls
        del self.attack
        del self.stats
        del self.X
        del self.y
        del self.config
        del self.folds
        del self.mask_cols
        del self.attr_ranges
        gc.collect()
