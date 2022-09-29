from abscls import AbsClassifierInstance
from attack import AbsAttack
from tree import DecisionTree
from xgb import XGBClassifier


class ClsLoader:
    XGBOOST = 0
    DECISION_TREE = 1

    @staticmethod
    def load(kind) -> AbsClassifierInstance:
        if kind == ClsLoader.DECISION_TREE:
            return DecisionTree()
        else:
            return XGBClassifier()


class AttackLoader:
    HOPSKIP = 0
    ZOO = 1

    @staticmethod
    def load(kind) -> AbsAttack:
        if kind == AttackLoader.HOPSKIP:
            return None
        else:
            return None
