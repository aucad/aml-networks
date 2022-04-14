"""
Simple adversarial example using ART with scikit-learn and applying
Zeroth-Order Optimization (ZOO) Attack.

The black-box zeroth-order optimization attack from Pin-Yu Chen et
al. (2018). This attack is a variant of the C&W attack which uses
ADAM coordinate descent to perform numerical estimation of gradients.

Adapted from:

https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/
4a65fd5e9cdc1f9a84fbb2c1a3ba42997fcfa3c6/notebooks/
classifier_scikitlearn_DecisionTreeClassifier.ipynb

Usage:

```
python src/adv_iris.py
```

Optionally pass as first positional argument the number of classes,
which is an integer [2,3], e.g.:

```
python src/adv_iris.py 3
```
"""

import numpy as np
from sys import argv

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt

from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import ZooAttack


def get_adversarial_examples(x_train, y_train):
    # Create and fit DecisionTreeClassifier
    model = DecisionTreeClassifier()
    model.fit(X=x_train, y=y_train)

    # Create ART classifier for scikit-learn DecisionTreeClassifier
    art_classifier = SklearnClassifier(model=model)

    # Create ART Zeroth Order Optimization attack
    zoo = ZooAttack(classifier=art_classifier, confidence=0.0,
                    targeted=False, learning_rate=1e-1, max_iter=20,
                    binary_search_steps=10, initial_const=1e-3,
                    abort_early=True, use_resize=False,
                    use_importance=False, nb_parallel=1, batch_size=1,
                    variable_h=0.2)

    # Generate adversarial samples with ART Zeroth Order
    # Optimization attack
    x_train_adv = zoo.generate(x_train)

    return x_train_adv, model


def get_data(num_classes):
    x_train, y_train = load_iris(return_X_y=True)
    x_train = x_train[y_train < num_classes][:, [0, 1]]
    y_train = y_train[y_train < num_classes]
    x_train[:, 0][y_train == 0] *= 2
    x_train[:, 1][y_train == 2] *= 2
    x_train[:, 0][y_train == 0] -= 3
    x_train[:, 1][y_train == 2] -= 2

    x_train[:, 0] = (x_train[:, 0] - 4) / (9 - 4)
    x_train[:, 1] = (x_train[:, 1] - 1) / (6 - 1)

    return x_train, y_train


def plot_results(model, x_train, y_train, x_train_adv, num_classes):
    fig, axs = plt.subplots(
        1, num_classes, figsize=(num_classes * 5, 5))

    colors = ['orange', 'blue', 'green']

    for i_class in range(num_classes):

        # Plot difference vectors
        for i in range(y_train[y_train == i_class].shape[0]):
            x_1_0 = x_train[y_train == i_class][i, 0]
            x_1_1 = x_train[y_train == i_class][i, 1]
            x_2_0 = x_train_adv[y_train == i_class][i, 0]
            x_2_1 = x_train_adv[y_train == i_class][i, 1]
            if x_1_0 != x_2_0 or x_1_1 != x_2_1:
                axs[i_class].plot([x_1_0, x_2_0], [x_1_1, x_2_1],
                                  c='black', zorder=1)

        # Plot benign samples
        for i_class_2 in range(num_classes):
            axs[i_class].scatter(x_train[y_train == i_class_2][:, 0],
                                 x_train[y_train == i_class_2][:, 1],
                                 s=20,
                                 zorder=2, c=colors[i_class_2])
        axs[i_class].set_aspect('equal', adjustable='box')

        # Show predicted probability as contour plot
        h = .01
        x_min, x_max = 0, 1
        y_min, y_max = 0, 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        Z_proba = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        Z_proba = Z_proba[:, i_class].reshape(xx.shape)
        im = axs[i_class].contourf(
            xx, yy, Z_proba,
            levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                    0.6, 0.7, 0.8, 0.9, 1.0],
            vmin=0, vmax=1)
        if i_class == num_classes - 1:
            cax = fig.add_axes([0.95, 0.2, 0.025, 0.6])
            plt.colorbar(im, ax=axs[i_class], cax=cax)

        # Plot adversarial samples
        for i in range(y_train[y_train == i_class].shape[0]):
            x_1_0 = x_train[y_train == i_class][i, 0]
            x_1_1 = x_train[y_train == i_class][i, 1]
            x_2_0 = x_train_adv[y_train == i_class][i, 0]
            x_2_1 = x_train_adv[y_train == i_class][i, 1]
            if x_1_0 != x_2_0 or x_1_1 != x_2_1:
                axs[i_class].scatter(x_2_0, x_2_1, zorder=2, c='red',
                                     marker='X')
        axs[i_class].set_xlim((x_min, x_max))
        axs[i_class].set_ylim((y_min, y_max))

        axs[i_class].set_title('class ' + str(i_class))
        axs[i_class].set_xlabel('feature 1')
        axs[i_class].set_ylabel('feature 2')


def iris(num_classes, fname):
    num_classes = min(3, max(2, num_classes))
    x_train, y_train = get_data(num_classes=num_classes)
    x_train_adv, model = get_adversarial_examples(x_train, y_train)
    plot_results(model, x_train, y_train, x_train_adv, num_classes)
    plt.savefig(fname)
    plt.show()


if __name__ == '__main__':
    class_count = int(argv[1]) if len(argv) > 1 else 2
    image_name = f'adversarial/iris_{class_count}.png'
    iris(class_count, image_name)
