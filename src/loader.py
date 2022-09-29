from cls import AbsClassifierInstance
from attack import AbsAttack
from hopskip import HopSkip
from zoo import Zoo
from tree import DecisionTree
from xgb import XGBClassifier


class ClsLoader:
    XGBOOST = 0
    DECISION_TREE = 1

    @staticmethod
    def load(kind=None) -> AbsClassifierInstance:
        if kind == ClsLoader.DECISION_TREE:
            return DecisionTree()
        else:
            return XGBClassifier()


class AttackLoader:
    HOP_SKIP = 0
    ZOO = 1

    @staticmethod
    def load(kind) -> AbsAttack:
        if kind == AttackLoader.HOP_SKIP:
            return HopSkip()
        else:
            return Zoo()
