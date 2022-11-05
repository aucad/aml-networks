import logging
from collections import namedtuple

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import KFold

from src import AbsClassifierInstance, AbsAttack, Validator, \
    HopSkip, Zoo, DecisionTree, XGBClassifier, Show

logger = logging.getLogger(__name__)


class ClsLoader:
    """Load selected classifier."""
    XGBOOST = 'xgb'
    DECISION_TREE = 'tree'

    @staticmethod
    def init(kind, *args) -> AbsClassifierInstance:
        if kind == ClsLoader.DECISION_TREE:
            return DecisionTree(*args)
        else:
            return XGBClassifier(*args)


class AttackLoader:
    """Load selected attack mode."""
    HOP_SKIP = 'hop'
    ZOO = 'zoo'

    @staticmethod
    def load(kind, *args) -> AbsAttack:
        if kind == AttackLoader.HOP_SKIP:
            return HopSkip(*args)
        else:
            return Zoo(*args)


class Experiment:
    DEFAULT_DS = 'data/CTU-1-1.csv'
    DEFAULT_CLS = ClsLoader.XGBOOST
    CLASSIFIERS = [ClsLoader.DECISION_TREE, ClsLoader.XGBOOST]
    ATTACKS = [AttackLoader.HOP_SKIP, AttackLoader.ZOO]
    VALIDATORS = [Validator.NB15, Validator.IOT23]

    def __init__(self, uuid, **kwargs):
        self.cls = None
        self.attack = None
        self.start_time = 0
        self.end_time = 0
        self.uuid = uuid
        self.attrs = []
        self.X = None
        self.y = None
        self.folds = None
        self.stats = Experiment.Stats()
        config_keys = ",".join(kwargs.keys())
        self.config = namedtuple('exp', config_keys)(**kwargs)

    class Stats:

        def __init__(self):
            self.accuracy = []
            self.precision = []
            self.recall = []
            self.f_score = []
            self.n_attack_records = []
            self.n_evasions = []
            self.n_valid = []

        def append_attack(self, attack: AbsAttack):
            self.n_evasions.append(attack.n_evasions)
            self.n_valid.append(attack.n_valid)
            self.n_attack_records.append(attack.n_records)

        def append_cls(self, cls: AbsClassifierInstance):
            self.accuracy.append(cls.accuracy)
            self.precision.append(cls.precision)
            self.recall.append(cls.recall)
            self.f_score.append(cls.f_score)

        @staticmethod
        def calc_average(arr):
            return 0 if len(arr) == 0 else sum(arr) / len(arr)

        @property
        def avg_accuracy(self) -> float:
            return self.calc_average(self.accuracy)

        @property
        def avg_precision(self) -> float:
            return self.calc_average(self.precision)

        @property
        def avg_recall(self) -> float:
            return self.calc_average(self.recall)

        @property
        def avg_f_score(self) -> float:
            return self.calc_average(self.f_score)

        @property
        def avg_n_evasions(self) -> float:
            return self.calc_average(self.n_evasions)

        @property
        def avg_n_valid(self) -> float:
            return self.calc_average(self.n_valid)

        @property
        def avg_n_records(self) -> float:
            return self.calc_average(self.n_attack_records)

        @property
        def ev_percent(self) -> float:
            if self.avg_n_records == 0:
                return 0
            return 100 * self.avg_n_evasions / self.avg_n_records

        @property
        def vd_percent(self) -> float:
            if self.avg_n_evasions == 0:
                return 0
            return 100 * self.avg_n_valid / self.avg_n_evasions

    @property
    def n_records(self):
        return len(self.X)

    @property
    def n_attr(self):
        return len(self.attrs)

    @property
    def duration(self):
        seconds = round((self.end_time - self.start_time) / 1e9, 2)
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
        self.load_csv(config.dataset, config.kfolds)
        cls_args = (config.cls, config.out, self.attrs, self.X,
                    self.y, config.dataset, config.robust)
        atk_args = (config.attack, False, False,
                    config.validator, self.uuid,
                    config.capture, config.iter)

        self.cls = ClsLoader.init(*cls_args)
        self.attack = AttackLoader.load(*atk_args) \
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
        Show('K-folds', self.config.kfolds)
        Show('Attack max iter', self.attack.max_iter)
        Show('Mutable', ", ".join(self.cls.mutable_attrs))
        Show('Immutable', ", ".join(self.cls.immutable_attrs))

    def log_fold_result(self, fold_n):
        Show('Fold', fold_n)
        Show('Accuracy', f'{self.cls.accuracy * 100:.2f} %')
        Show('Precision', f'{self.cls.precision * 100:.2f} %')
        Show('Recall', f'{self.cls.recall * 100:.2f} %')
        Show('F-score', f'{self.cls.f_score * 100:.2f} %')
        Show('Total evasions',
             f'{self.attack.n_evasions} of {self.attack.n_records} '
             f'- {self.attack.evasion_success:.1f} %')
        if self.attack.validator_kind:
            Show('Valid evasions',
                 f'{self.attack.n_valid} of {self.attack.n_evasions} '
                 f'- {self.attack.validation_success:.1f} %')
        if self.attack.n_evasions > 0:
            ls = self.attack.printable_label_stats()
            Show('Class labels', '\n'.join(ls))
        if self.attack.n_evasions != self.attack.n_valid:
            Validator.dump_reasons(self.attack.validation_reasons)
        min_e, max_e = self.attack.calculate_error()
        Show('L-norm', f'{min_e:.6f} - {max_e:.6f}')

    def log_experiment_result(self):
        sh = lambda x, y: Show(f'Avg. {x}', f'{(y * 100):.2f} %')
        e = self.stats.avg_n_evasions
        t = self.stats.avg_n_records
        v = self.stats.avg_n_valid
        p = self.stats.ev_percent
        q = self.stats.vd_percent
        min_, sec = self.duration

        Show('=' * 52, '')
        sh('Accuracy', self.stats.avg_accuracy)
        sh('Precision', self.stats.avg_precision)
        sh('Recall', self.stats.avg_recall)
        sh('F-score', self.stats.avg_f_score)
        Show('Total evasions', f'{e} of {t} - {p:.1f} %')
        if self.config.validator and e > 0:
            Show('Valid evasions', f'{v} of {e} - {q:.1f} %')
        Show('Time', f'{min_} min {sec:.0f} s')
