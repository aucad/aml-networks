import logging
from collections import namedtuple

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import KFold

from src import AbsClassifierInstance, Validator, \
    AbsAttack, HopSkip, Zoo, \
    DecisionTree, XGBClassifier, Show

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
    CLASSIFIERS = [ClsLoader.DECISION_TREE, ClsLoader.XGBOOST]
    DEFAULT_CLS = ClsLoader.XGBOOST
    ATTACKS = [AttackLoader.HOP_SKIP, AttackLoader.ZOO]
    VALIDATORS = [Validator.NB15, Validator.IOT23]

    def __init__(self, uuid, **kwargs):
        print(uuid)
        self.start_time = 0
        self.end_time = 0
        self.uuid = uuid
        self.attrs = []
        self.config = namedtuple('exp', ",".join(kwargs.keys()))(**kwargs)
        self.X = None
        self.y = None
        self.folds = None

    @property
    def n_records(self):
        return len(self.X)

    @property
    def n_attr(self):
        return len(self.attrs)

    def load_csv(self, ds_path, n_splits):
        df = pd.read_csv(ds_path).fillna(0)
        self.attrs = [col for col in df.columns]
        self.X = (np.array(df)[:, :-1])
        self.y = np.array(df)[:, -1].astype(int).flatten()
        self.folds = [(tr_i, ts_i) for tr_i, ts_i
                      in KFold(n_splits=n_splits).split(self.X)]

    def do_fold(self, cls, attack, fold_num, fold_indices):
        cls.reset() \
            .load(self.X.copy(), self.y.copy(), *fold_indices, fold_num) \
            .train()

        if attack:
            attack.reset().set_cls(cls).run(max_iter=self.config.iter)

    def run(self):
        config = self.config
        self.load_csv(config.dataset, config.kfolds)
        cls_args = (config.cls, config.out, self.attrs, self.X, self.y,
                    config.dataset, config.robust)
        atk_args = (config.attack, False, config.plot,
                    config.validator, config.dataset,
                    self.uuid, config.save_rec)
        self.start_time = time.time_ns()
        averages = []

        cls = ClsLoader.init(*cls_args)
        attack = AttackLoader.load(*atk_args) if config.attack else None
        self.log_experiment_setup()

        for i, fold in enumerate(self.folds):
            self.do_fold(cls, attack, i + 1, fold)
            averages.append(list(cls.stats) + list(attack.stats))
            if config.plot:
                cls.plot()

        self.log_experiment_result(averages)

    def log_experiment_setup(self):
        Show('Dataset', self.config.dataset)
        Show('Record count', self.n_records)
        Show('Attributes', self.n_attr)

    def log_experiment_result(self, averages):
        Show('=' * 52, '')
        a = list(np.mean(np.array(averages), axis=0))
        v, e, t = a[-3:]
        print('AVERAGES', '')
        print('Classifier', '')
        Show('Accuracy', f'{a[0] * 100:.2f} %')
        Show('Precision', f'{a[1] * 100:.2f} %')
        Show('Recall', f'{a[2] * 100:.2f} %')
        Show('F-score', f'{a[3] * 100:.2f} %')
        print('Attack', '')
        Show('Total evasions', f'{e} of {t} - {(100 * (e / t)):.1f} %')
        if self.config.validator and e > 0:
            Show('Valid evasions',
                 f'{v} of {e} - {100 * (v / e):.1f} %')

        end_time = time.time_ns()
        seconds = round((end_time - self.start_time) / 1e9, 2)
        minutes = int(seconds // 60)
        seconds = seconds - (minutes * 60)
        Show('Time', f'{minutes} min {seconds:.2f} s')
