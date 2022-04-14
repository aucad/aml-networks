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
python src/adv_mnist.py
```
"""

import numpy as np

from sys import path
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import ZooAttack
from art.utils import load_mnist


def mnist_attack(out_dir='adversarial', save_images=False):
    (x_train, y_train), (x_test, y_test), min_, max_ = load_mnist()

    n_samples_train = x_train.shape[0]
    n_features_train = x_train.shape[1] * x_train.shape[2] * \
                       x_train.shape[3]
    n_samples_test = x_test.shape[0]
    n_features_test = x_test.shape[1] * x_test.shape[2] * x_test.shape[
        3]

    x_train = x_train.reshape(n_samples_train, n_features_train)
    x_test = x_test.reshape(n_samples_test, n_features_test)

    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)

    n_samples_max = 200
    x_train = x_train[0:n_samples_max]
    y_train = y_train[0:n_samples_max]
    x_test = x_test[0:n_samples_max]
    y_test = y_test[0:n_samples_max]

    # Train DecisionTreeClassifier classifier
    model = DecisionTreeClassifier(
        criterion='gini', splitter='best',
        max_depth=None, min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None, max_leaf_nodes=50,
        min_impurity_decrease=0.0,
        class_weight=None)
    model.fit(X=x_train, y=y_train)

    # Create and apply Zeroth Order Optimization Attack with ART
    art_classifier = SklearnClassifier(model=model)

    zoo = ZooAttack(
        classifier=art_classifier, confidence=0.0, targeted=False,
        learning_rate=1e-1, max_iter=100, binary_search_steps=20,
        initial_const=1e-3, abort_early=True, use_resize=False,
        use_importance=False, nb_parallel=10, batch_size=1,
        variable_h=0.25)

    x_train_adv = zoo.generate(x_train)
    x_test_adv = zoo.generate(x_test)

    # Evaluate DecisionTreeClassifier on benign and adversarial samples
    score = model.score(x_train, y_train)
    print("Benign Training Score: %.4f" % score)
    plt.matshow(x_train[0, :].reshape((28, 28)))
    plt.clim(0, 1)
    if save_images:
        plt.savefig(path.join(out_dir, 'benign_train.png'))

    prediction = model.predict(x_train[0:1, :])[0]
    print("Benign Training Predicted Label: %i" % prediction)

    score = model.score(x_train_adv, y_train)
    print("Adversarial Training Score: %.4f" % score)

    plt.matshow(x_train_adv[0, :].reshape((28, 28)))
    plt.clim(0, 1)
    if save_images:
        plt.savefig(path.join(out_dir, '/adv_train.png'))

    prediction = model.predict(x_train_adv[0:1, :])[0]
    print("Adversarial Training Predicted Label: %i" % prediction)

    score = model.score(x_test, y_test)
    print("Benign Test Score: %.4f" % score)

    plt.matshow(x_test[0, :].reshape((28, 28)))
    plt.clim(0, 1)
    if save_images:
        plt.savefig(path.join(out_dir, 'benign_test.png'))

    prediction = model.predict(x_test[0:1, :])[0]
    print("Benign Test Predicted Label: %i" % prediction)

    score = model.score(x_test_adv, y_test)
    print("Adversarial Test Score: %.4f" % score)

    plt.matshow(x_test_adv[0, :].reshape((28, 28)))
    plt.clim(0, 1)
    if save_images:
        plt.savefig(path.join(out_dir, 'adv_test.png'))

    prediction = model.predict(x_test_adv[0:1, :])[0]
    print("Adversarial Test Predicted Label: %i" % prediction)


if __name__ == '__main__':
    mnist_attack()
