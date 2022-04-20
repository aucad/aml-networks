"""
Simple adversarial example using ART with scikit-learn and applying
Zeroth-Order Optimization (ZOO) Attack.

The black-box zeroth-order optimization attack from Pin-Yu Chen et
al. (2018). This attack is a variant of the C&W attack which uses
ADAM coordinate descent to perform numerical estimation of gradients.

Usage:

```
python src/adv_iot.py
```

"""

import numpy as np
from art.attacks.evasion import ZooAttack
from art.estimators.classification import SklearnClassifier
from matplotlib import pyplot as plt

from tree import train_tree


def adversarial_iot():
    colors = ['blue', 'green']

    model, x_train, y_train = train_tree(False)
    n_classes = 2

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
    fig, axs = plt.subplots(1, n_classes, figsize=(n_classes * 5, 5))

    for i, class_label in enumerate(y_train[0:2]):

        # Plot difference vectors
        for j in range(y_train[y_train == i].shape[0]):
            x_1_0 = x_train[y_train == i][j, 0]
            x_1_1 = x_train[y_train == i][j, 1]
            x_2_0 = x_train_adv[y_train == i][j, 0]
            x_2_1 = x_train_adv[y_train == i][j, 1]
            if x_1_0 != x_2_0 or x_1_1 != x_2_1:
                axs[i].plot([x_1_0, x_2_0], [x_1_1, x_2_1],
                            c='black', zorder=1)

        # Plot benign samples
        for i_class_2 in range(n_classes):
            axs[i].scatter(
                x_train[y_train == i_class_2][:, 0],
                x_train[y_train == i_class_2][:, 1],
                s=20, zorder=2, c=colors[i_class_2])
        axs[i].set_aspect('equal', adjustable='box')

        x_min, x_max = 0, 1
        y_min, y_max = 0, 1

        # Plot adversarial samples
        for j in range(y_train[y_train == i].shape[0]):
            x_1_0 = x_train[y_train == i][j, 0]
            x_1_1 = x_train[y_train == i][j, 1]
            x_2_0 = x_train_adv[y_train == i][j, 0]
            x_2_1 = x_train_adv[y_train == i][j, 1]
            if x_1_0 != x_2_0 or x_1_1 != x_2_1:
                axs[i].scatter(x_2_0, x_2_1, zorder=2,
                               c='red', marker='X')

        axs[i].set_xlim((x_min, x_max))
        axs[i].set_ylim((y_min, y_max))
        axs[i].set_title('class ' + str(i))
        axs[i].set_xlabel('feature 1')
        axs[i].set_ylabel('feature 2')

    plt.show()


if __name__ == '__main__':
    adversarial_iot()
