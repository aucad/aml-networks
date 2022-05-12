"""
Simple adversarial example using ART with scikit-learn and applying
attribute inference attack.

In order to mount a successful attribute inference attack,
the attacked feature must be categorical, and with a relatively small
number of possible values (preferably binary, but should at least be
less then the number of label classes).

Inspired by this example:

https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/
084e9769fc84934f77ec600ced5452a0e9baa60f/notebooks/
attack_attribute_inference.ipynb

Usage:

```
python src/inf_iot.py
```
"""

import warnings

warnings.filterwarnings("ignore")  # ignore import warnings

import numpy as np
from art.attacks.inference.attribute_inference \
    import AttributeInferenceBlackBox
from art.estimators.classification.scikitlearn \
    import ScikitlearnDecisionTreeClassifier
from art.attacks.inference.attribute_inference \
    import AttributeInferenceWhiteBoxLifestyleDecisionTree
from art.attacks.inference.attribute_inference \
    import AttributeInferenceWhiteBoxDecisionTree

from tree import train_tree
from utility import color_text as c


def black_box(classifier, x_train, attr_index, label=None):
    """Trains an additional classifier (called the attack model) to
    predict the attacked feature's value from the remaining n-1 features
    as well as the original (attacked) model's predictions."""

    values = [0, 1]

    # Train attack model
    attack_train_ratio = 0.5
    attack_train_size = int(len(x_train) * attack_train_ratio)
    attack_x_train = x_train[:attack_train_size]
    attack_x_test = x_train[attack_train_size:]

    attack_feature = attr_index

    attack_x_test_predictions = np.array(
        [np.argmax(arr) for arr in
         classifier.predict(attack_x_test)]).reshape(-1, 1)

    # only attacked feature
    attack_x_test_feature = \
        attack_x_test[:, attack_feature].copy().reshape(-1, 1)

    # training data without attacked feature
    attack_x_test = np.delete(attack_x_test, attack_feature, 1)

    bb_attack = AttributeInferenceBlackBox(
        classifier, attack_feature=attack_feature)

    # train attack model
    bb_attack.fit(attack_x_train)

    # Infer sensitive feature and check accuracy
    # get inferred values

    # infer the attribute values
    inferred_train_bb = bb_attack.infer(
        attack_x_test,
        pred=attack_x_test_predictions,
        values=values)

    print("Inferred black box data: \n", inferred_train_bb)

    # check accuracy
    acc = np.sum(inferred_train_bb ==
                 np.around(attack_x_test_feature, decimals=8)
                 .reshape(1, -1)) / len(inferred_train_bb)
    print("Blackbox accuracy " + f'({label}):' if label else ':',
          c(f'{acc * 100:.2f} %'))


def white_box_one(classifier, x_train, attr_index, label=None):
    """These two attacks do not train any additional model, they simply
    use additional information coded within the attacked decision tree
    model to compute the probability of each value of the attacked
    feature and outputs the value with the highest probability."""
    wb_attack = AttributeInferenceWhiteBoxLifestyleDecisionTree(
        classifier, attack_feature=attr_index)

    values = [0, 1]
    priors = [3465 / 5183, 1718 / 5183]
    attack_train_ratio = 0.5
    attack_train_size = int(len(x_train) * attack_train_ratio)
    attack_x_test = x_train[attack_train_size:]

    attack_x_test_predictions = np.array(
        [np.argmax(arr) for arr in
         classifier.predict(attack_x_test)]).reshape(-1, 1)

    # only attacked feature
    attack_x_test_feature = \
        attack_x_test[:, attr_index].copy().reshape(-1, 1)

    # get inferred values
    inferred_train_wb1 = wb_attack.infer(
        attack_x_test, attack_x_test_predictions,
        values=values, priors=priors)

    # check accuracy
    acc = np.sum(inferred_train_wb1 == np.around(
        attack_x_test_feature, decimals=8).reshape(1, -1)) / len(
        inferred_train_wb1)
    print('White-box I accuracy' + f'({label}):' if label else ':',
          c(f'{acc * 100:.2f} %'))


def white_box_two(classifier, x_train, attr_index, label=None):
    """These two attacks do not train any additional model, they simply
    use additional information coded within the attacked decision tree
    model to compute the probability of each value of the attacked
    feature and outputs the value with the highest probability."""
    wb2_attack = AttributeInferenceWhiteBoxDecisionTree(
        classifier, attack_feature=attr_index)

    values = [0, 1]
    priors = [3465 / 5183, 1718 / 5183]
    attack_train_ratio = 0.5
    attack_train_size = int(len(x_train) * attack_train_ratio)
    attack_x_test = x_train[attack_train_size:]

    attack_x_test_predictions = np.array(
        [np.argmax(arr) for arr in
         classifier.predict(attack_x_test)]).reshape(-1, 1)

    # only attacked feature
    attack_x_test_feature = \
        attack_x_test[:, attr_index].copy().reshape(-1, 1)

    # get inferred values
    inferred_train_wb2 = wb2_attack.infer(
        attack_x_test, attack_x_test_predictions,
        values=values, priors=priors)

    # check accuracy
    acc = np.sum(inferred_train_wb2 == np.around(
        attack_x_test_feature, decimals=8).reshape(1, -1)) / len(
        inferred_train_wb2)
    print('White-box II accuracy' + f'({label}):' if label else ':',
          c(f'{acc * 100:.2f} %'))


def attr_inference():
    """Perform various attribute inference attacks."""

    # load decision tree model and data
    model, x_train, y_train, ATTR, x_test, y_test = \
        train_tree(False, test_set=0.25)

    art_classifier = ScikitlearnDecisionTreeClassifier(model)
    acc = model.score(x_test, y_test)
    print('Base model accuracy: ', c(f'{acc * 100:.2f} %'))

    black_box(art_classifier, x_train, 0, 'proto=udp')
    black_box(art_classifier, x_train, 4, 'conn_state=SF')

    # white_box_one(art_classifier, x_train, 0, 'proto=udp')
    # white_box_one(art_classifier, x_train, 4, 'conn_state=SF')


if __name__ == '__main__':
    attr_inference()
