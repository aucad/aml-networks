"""
Simple adversarial example using ART with scikit-learn and applying
Zeroth-Order Optimization (ZOO) Attack.

The black-box zeroth-order optimization attack from Pin-Yu Chen et
al. (2018). This attack is a variant of the C&W attack which uses
ADAM coordinate descent to perform numerical estimation of gradients.

MNIST contains images of numbers. The goal of this attack is to distort
image such that model will predict a numeric value that is different
from the prediction for original unmodified image.

Adapted from:

https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/
4a65fd5e9cdc1f9a84fbb2c1a3ba42997fcfa3c6/notebooks/
classifier_scikitlearn_DecisionTreeClassifier.ipynb

Usage:

```
python src/examples/zoo_mnist.py --help
```
"""

from argparse import ArgumentParser
from os import path

import numpy as np
from art.attacks.evasion import ZooAttack
from art.estimators.classification import SklearnClassifier
from art.utils import load_mnist
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier


def load_model(x_train, y_train):
    """Initialize decision tree classifier."""
    model = DecisionTreeClassifier(
        criterion='gini',
        splitter='best',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=50,
        min_impurity_decrease=0.0,
        class_weight=None)
    model.fit(X=x_train, y=y_train)
    return model


def get_data(n_samples_max):
    """Load MNIST dataset."""
    (x_train, y_train), (x_test, y_test), min_, max_ = load_mnist()

    n_samples_train = x_train.shape[0]
    n_features_train = x_train.shape[1] * \
                       x_train.shape[2] * \
                       x_train.shape[3]

    n_samples_test = x_test.shape[0]
    n_features_test = x_test.shape[1] * \
                      x_test.shape[2] * \
                      x_test.shape[3]

    x_train = x_train.reshape(n_samples_train, n_features_train)
    x_test = x_test.reshape(n_samples_test, n_features_test)
    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)

    x_train = x_train[0:n_samples_max]
    y_train = y_train[0:n_samples_max]
    x_test = x_test[0:n_samples_max]
    y_test = y_test[0:n_samples_max]
    return (x_train, y_train), (x_test, y_test)


def plot(out_dir, X, label):
    """Generate and save plot."""
    img = label.replace(' ', '_').lower()
    plt.matshow(X[0, :].reshape((28, 28)))
    plt.clim(0, 1)
    plt.savefig(path.join(out_dir, f'{img}.png'))


def evaluate(model, X, y, label):
    """Display score and prediction."""
    score = model.score(X, y),
    prediction = model.predict(X[0:1, :])[0]
    print(label, 'score:', "%.4f" % score)
    print(label, 'prediction label:', prediction)
    return True


def mnist_attack(n, out_dir):
    """Carry out the attack."""

    # train the classifier
    (x_train, y_train), (x_test, y_test) = get_data(n)
    model = load_model(x_train, y_train)

    # create and apply ZOO attack with ART
    art_classifier = SklearnClassifier(model=model)

    zoo = ZooAttack(
        classifier=art_classifier,
        confidence=0.0, targeted=False,
        learning_rate=1e-1, max_iter=100,
        binary_search_steps=20, initial_const=1e-3,
        abort_early=True, use_resize=False,
        use_importance=False, nb_parallel=10,
        batch_size=1, variable_h=0.25)

    x_train_adv = zoo.generate(x_train)
    x_test_adv = zoo.generate(x_test)

    eval_ = lambda x, y, label: \
        evaluate(model, x, y, label) and \
        plot(out_dir, x, label)

    # evaluate classifier on benign and adversarial samples
    eval_(x_train, y_train, 'Benign Train')
    eval_(x_train_adv, y_train, 'Adv Train')
    eval_(x_test, y_test, 'Benign Test')
    eval_(x_test_adv, y_test, 'Adv Test')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--out',
        dest='out',
        action='store',
        default='adversarial',
        help='output directory')
    parser.add_argument(
        '--n',
        default=200,
        help='input size',
        type=int)

    args = parser.parse_args()
    mnist_attack(args.n, args.out)
