from abscls import AbsClassifierInstance
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
