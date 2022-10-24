import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src import AbsClassifierInstance, AbsAttack, \
    HopSkip, Zoo, DecisionTree, XGBClassifier
from src.utility import BaseUtil


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


class DatasetLoader:

    @staticmethod
    def load_csv(ds_path, n_splits):
        df = pd.read_csv(ds_path).fillna(0)
        attrs = [col for col in df.columns]
        X = (np.array(df)[:, :-1])
        y = np.array(df)[:, -1].astype(int).flatten()
        folds = [(tr_i, ts_i)
                 for tr_i, ts_i in KFold(n_splits=n_splits).split(X)]
        BaseUtil.show('Read dataset', ds_path)
        BaseUtil.show('Attributes', len(attrs))
        BaseUtil.show('Training size', len(folds[0][0]))
        BaseUtil.show('Test size', len(folds[0][1]))
        return X, y, attrs, folds
