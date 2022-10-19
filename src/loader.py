from src import AbsClassifierInstance, AbsAttack, \
    HopSkip, Zoo, DecisionTree, XGBClassifier


class ClsLoader:
    """Load selected classifier."""
    XGBOOST = 'xgb'
    DECISION_TREE = 'tree'

    @staticmethod
    def load(out_dir, kind=None) -> AbsClassifierInstance:
        if kind == ClsLoader.DECISION_TREE:
            return DecisionTree(out_dir)
        else:
            return XGBClassifier(out_dir)


class AttackLoader:
    """Load selected attack mode."""
    HOP_SKIP = 'hop'
    ZOO = 'zoo'

    @staticmethod
    def load(kind, iterated, plot) -> AbsAttack:
        args = (iterated, plot)
        if kind == AttackLoader.HOP_SKIP:
            return HopSkip(*args)
        else:
            return Zoo(*args)
