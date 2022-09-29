from cls import AbsClassifierInstance
from attack import AbsAttack
from hopskip import HopSkip
from zoo import Zoo
from tree import DecisionTree
from xgb import XGBClassifier


class ClsLoader:
    """Load selected classifier."""
    XGBOOST = 'xgb'
    DECISION_TREE = 'tree'

    @staticmethod
    def load(kind=None) -> AbsClassifierInstance:
        if kind == ClsLoader.DECISION_TREE:
            return DecisionTree()
        else:
            return XGBClassifier()


class AttackLoader:
    """Load selected attack mode."""
    HOP_SKIP = 'hop'
    ZOO = 'zoo'

    @staticmethod
    def load(kind) -> AbsAttack:
        if kind == AttackLoader.HOP_SKIP:
            return HopSkip()
        else:
            return Zoo()
