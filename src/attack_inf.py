"""
Simple adversarial example using ART with scikit-learn and applying
attribute inference attack using decision tree classifier.

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
python src/attack_inf.py
```
"""
import warnings

warnings.filterwarnings("ignore")  # ignore import warnings

import numpy as np
from art.attacks.inference.attribute_inference \
    import AttributeInferenceBaseline, \
    AttributeInferenceBlackBox, \
    AttributeInferenceMembership, \
    AttributeInferenceWhiteBoxLifestyleDecisionTree, \
    AttributeInferenceWhiteBoxDecisionTree
from art.estimators.classification.scikitlearn import \
    ScikitlearnDecisionTreeClassifier
from art.attacks.inference.membership_inference \
    import MembershipInferenceBlackBox

from utility import color_text as c, binary_attributes


def evaluate(attack_name, x_test_feat, inferred_train):
    """Display inference performance metrics."""
    if inferred_train is None:
        return

    actual = np.around(x_test_feat, decimals=8)
    a, p, r = calc_performance(inferred_train, actual, 0)

    print(f'{attack_name} attack '.ljust(30, '-'), end=' ')
    print(f'Accuracy:', c(f'{a * 100:.2f} %'), end=' ')
    print(f'Precision:', c(f'{p * 100:.2f} %'), end=' ')
    print(f'Recall:', c(f'{r * 100:.2f} %'))


def calc_performance(predicted, actual, pos_val):
    """Calculate performance metrics."""
    acc = np.sum(predicted == actual.reshape(1, -1)) / len(predicted)
    score, num_pos_predicted, num_pos_actual = 0, 0, 0

    for i in range(len(predicted)):
        if predicted[i] == pos_val:
            num_pos_predicted += 1
        if actual[i] == pos_val:
            num_pos_actual += 1
        if predicted[i] == actual[i] and predicted[i] == pos_val:
            score += 1

    precision = 1 if num_pos_predicted == 0 else \
        score / num_pos_predicted

    recall = 1 if num_pos_actual == 0 else \
        score / num_pos_actual

    return acc, precision, recall


def baseline(feat, x_train, x_test, values):
    """Implementation of a baseline attribute inference, not using a
    model. The idea is to train a simple neural network to learn the
    attacked feature from the rest of the features. Should be used to
    compare with other attribute inference results."""
    baseline_attack = AttributeInferenceBaseline(attack_feature=feat)
    baseline_attack.fit(x_train)

    return baseline_attack.infer(x_test, values=values)


def black_box(cls, feat, x_train, x_test, predictions, values):
    """Implementation of a simple black-box attribute inference
    attack. The idea is to train a simple neural network to learn the
    attacked feature from the rest of the features and the model’s
    predictions. Assumes the availability of the attacked model’s
    predictions for the samples under attack, in addition to the rest
    of the feature values. If this is not available, the true class
    label of the samples may be used as a proxy."""
    bb_attack = AttributeInferenceBlackBox(cls, attack_feature=feat)
    bb_attack.fit(x_train)

    # infer the attribute values for sensitive feature
    return bb_attack.infer(x_test, pred=predictions, values=values)


def white_box_1(cls, feat, x_test, x_pred, values, priors):
    """A variation of the method proposed by of Fredrikson et al.
    Assumes the availability of the attacked model’s predictions for
    the samples under attack, in addition to access to the model itself
    and the rest of the feature values. If this is not available, the
    true class label of the samples may be used as a proxy. Also assumes
    that the attacked feature is discrete or categorical, with limited
    number of possible values. For example: a boolean feature.
    Paper link: https://dl.acm.org/doi/10.1145/2810103.2813677
    """
    if not isinstance(cls, ScikitlearnDecisionTreeClassifier):
        return None

    wb_attack = AttributeInferenceWhiteBoxDecisionTree(
        cls, attack_feature=feat)

    return wb_attack.infer(x_test, x_pred, values=values, priors=priors)


def white_box_2(cls, feat, x_test, x_pred, values, priors):
    """Implementation of Fredrikson et al. white box inference attack
    for decision trees. Assumes that the attacked feature is discrete
    or categorical, with limited number of possible values. For
    example: a boolean feature. Paper link:
    https://dl.acm.org/doi/10.1145/2810103.2813677
    """
    if not isinstance(cls, ScikitlearnDecisionTreeClassifier):
        return None

    wb_attack = AttributeInferenceWhiteBoxLifestyleDecisionTree(
        cls, attack_feature=feat)

    return wb_attack.infer(x_test, x_pred, values=values, priors=priors)


def membership(cls, feat, values, x_test, y_test, fit_data):
    """Implementation of an attribute inference attack that
    utilizes a membership inference attack. The idea is to find the
    target feature value that causes the membership inference attack
    to classify the sample as a member with the highest confidence."""

    # Membership inference Implementation of a learned black-box
    # membership inference attack. This implementation can use as
    # input to the learning process probabilities/logits or losses,
    # depending on the type of model and provided configuration.
    mem_attack = MembershipInferenceBlackBox(cls)
    mem_attack.fit(*fit_data)

    # Apply attribute attack
    attack = AttributeInferenceMembership(
        cls, mem_attack, attack_feature=feat)

    # infer values
    return attack.infer(x_test, y_test, values=values)


def infer(cls, feat, name, xtrain, ytrain, xtest, ytest):
    """Carry out inference attacks"""

    # training and test split
    attack_train_ratio = 0.5
    attack_train_size = int(len(xtrain) * attack_train_ratio)
    x_train = xtrain[:attack_train_size]
    x_test = xtrain[attack_train_size:]
    y_test = ytrain[attack_train_size:]
    ms_fit_data = (xtrain[:attack_train_size],
                   ytrain[:attack_train_size], xtest, ytest)

    predictions = np.array(
        [np.argmax(arr) for arr in cls.predict(x_test)]).reshape(-1, 1)

    # only attacked feature
    attack_feat = x_test[:, feat].copy().reshape(-1, 1)

    # training data without attacked feature
    x_test = np.delete(x_test, feat, 1)

    # possible attribute values and prior data distributions
    values = [0, 1]
    priors = [(attack_feat == v).sum() / len(x_test) for v in values]

    # carry out various inference attacks
    bl = baseline(feat, x_train, x_test, values)
    bb = black_box(cls, feat, x_train, x_test, predictions, values)
    w1 = white_box_1(cls, feat, x_test, predictions, values, priors)
    w2 = white_box_2(cls, feat, x_test, predictions, values, priors)
    ms = membership(cls, feat, values, x_test, y_test, ms_fit_data)

    # display results
    print(f'* Inference of attribute {name}:')
    evaluate("Baseline", attack_feat, bl)
    evaluate("Black box", attack_feat, bb)
    evaluate("White box 1", attack_feat, w1)
    evaluate("White box 2", attack_feat, w2)
    evaluate("Membership", attack_feat, ms)


def inference_attack(load_classifier, **cls_kwargs):
    """Perform various attribute inference attacks."""

    # load decision tree model and data
    classifier, model, attr_names, x_train, y_train, x_test, y_test = \
        load_classifier(**cls_kwargs)

    try:
        attack_attributes = list(set.intersection(
            set(binary_attributes(x_train)),
            set(binary_attributes(x_test))))
    except TypeError:
        attack_attributes = [0]

    for idx in attack_attributes:
        infer(classifier, idx, attr_names[idx],
              x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    from train_xg import train

    inference_attack(train, test_size=.25)
