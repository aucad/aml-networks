"""
Simple adversarial example using ART with scikit-learn and applying
Zeroth-Order Optimization (ZOO) Attack.

The black-box zeroth-order optimization attack from Pin-Yu Chen et
al. (2018). This attack is a variant of the C&W attack which uses
ADAM coordinate descent to perform numerical estimation of gradients.

"""

import warnings

warnings.filterwarnings('ignore')

from utility import color_text as c
from tree import train_tree


# import numpy as np
# from matplotlib import pyplot as plt
# from sklearn.tree import DecisionTreeClassifier
# from art.estimators.classification import SklearnClassifier
# from art.attacks.evasion import ZooAttack


def adversarial_iot():
    trained_model, X, y = train_tree(False)
    print(c(f'TODO: attack this tree...'))


if __name__ == '__main__':
    adversarial_iot()
