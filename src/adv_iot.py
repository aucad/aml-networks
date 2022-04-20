"""
Simple adversarial example using ART with scikit-learn and applying
Zeroth-Order Optimization (ZOO) Attack.

The black-box zeroth-order optimization attack from Pin-Yu Chen et
al. (2018). This attack is a variant of the C&W attack which uses
ADAM coordinate descent to perform numerical estimation of gradients.

"""

import numpy as np
from art.attacks.evasion import ZooAttack
from art.estimators.classification import SklearnClassifier

from tree import train_tree


def adversarial_iot():
    model, x_train, y_train = train_tree(False)

    # use numpy arrays
    x_train = np.array([np.array(xi) for xi in x_train])
    y_train = np.array(y_train)

    # Create ART Zeroth Order Optimization attack
    # using scikit-learn DecisionTreeClassifier
    zoo = ZooAttack(
        classifier=SklearnClassifier(model=model),
        confidence=0.0, targeted=False, learning_rate=1e-1,
        max_iter=20, binary_search_steps=10, initial_const=1e-3,
        abort_early=True, use_resize=False, use_importance=False,
        nb_parallel=1, batch_size=1, variable_h=0.2)

    x_train_adv = zoo.generate(x_train)

    score = model.score(x_train, y_train),
    prediction = model.predict(x_train[0:1, :])[0]
    print('train', 'score:', "%.4f" % score, prediction)

    score = model.score(x_train_adv, y_train),
    prediction = model.predict(x_train_adv[0:1, :])[0]
    print('adv  ', 'score:', "%.4f" % score, prediction)


if __name__ == '__main__':
    adversarial_iot()
