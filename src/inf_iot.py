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
    import AttributeInferenceBaseline, \
    AttributeInferenceBlackBox, \
    AttributeInferenceWhiteBoxLifestyleDecisionTree, \
    AttributeInferenceWhiteBoxDecisionTree
from art.estimators.classification.scikitlearn \
    import ScikitlearnDecisionTreeClassifier

from tree import train_tree
from utility import color_text as c


def calc_precision_recall(predicted, actual, positive_value=1):
    score = 0  # both predicted and actual are positive
    num_positive_predicted = 0  # predicted positive
    num_positive_actual = 0  # actual positive
    for i in range(len(predicted)):
        if predicted[i] == positive_value:
            num_positive_predicted += 1
        if actual[i] == positive_value:
            num_positive_actual += 1
        if predicted[i] == actual[i]:
            if predicted[i] == positive_value:
                score += 1

    if num_positive_predicted == 0:
        precision = 1
    else:
        # the fraction of predicted “Yes” responses that are correct
        precision = score / num_positive_predicted
    if num_positive_actual == 0:
        recall = 1
    else:
        # the fraction of “Yes” responses that are predicted correctly
        recall = score / num_positive_actual

    return f'{100 * precision:.2f} %  {100 * recall:.2f} %'


def black_box(classifier, x_train, attack_feature, label):
    """Trains an additional classifier (called the attack model) to
    predict the attacked feature's value from the remaining n-1 features
    as well as the original (attacked) model's predictions."""

    values = [0, 1]

    # training and test split
    attack_train_ratio = 0.5
    attack_train_size = int(len(x_train) * attack_train_ratio)
    attack_x_train = x_train[:attack_train_size]
    attack_x_test = x_train[attack_train_size:]

    attack_x_test_predictions = np.array(
        [np.argmax(arr) for arr in
         classifier.predict(attack_x_test)]).reshape(-1, 1)

    # only attacked feature
    attack_x_test_feat = \
        attack_x_test[:, attack_feature].copy().reshape(-1, 1)

    # training data without attacked feature
    attack_x_test = np.delete(attack_x_test, attack_feature, 1)

    bb_attack = AttributeInferenceBlackBox(
        classifier, attack_feature=attack_feature)

    # train attack model
    bb_attack.fit(attack_x_train)

    # infer the attribute values for sensitive feature
    inferred_train_bb = bb_attack.infer(
        attack_x_test, pred=attack_x_test_predictions, values=values)

    # check accuracy
    actual = np.around(attack_x_test_feat, decimals=8).reshape(1, -1)
    acc = np.sum(inferred_train_bb == actual) / len(inferred_train_bb)

    print(f'Black-box   ({label})', end=' ')
    print(f'Accuracy precision and recall:', c(f'{acc * 100:.2f} % '),
          c(calc_precision_recall(inferred_train_bb, np.around(
              attack_x_test_feat, decimals=8), positive_value=0)))


def white_box(classifier, x_train, attack_feature, label):
    """These two attacks do not train any additional model, they simply
    use additional information coded within the attacked decision tree
    model to compute the probability of each value of the attacked
    feature and outputs the value with the highest probability."""

    values = [0, 1]

    # training and test split
    attack_train_ratio = 0.80
    attack_train_size = int(len(x_train) * attack_train_ratio)
    attack_x_test = x_train[attack_train_size:]

    attack_x_test_predictions = np.array(
        [np.argmax(arr) for arr in
         classifier.predict(attack_x_test)]).reshape(-1, 1)

    # only attacked feature
    attack_x_test_feat = attack_x_test[:, attack_feature] \
        .copy().reshape(-1, 1)

    # training data without attacked feature
    attack_x_test = np.delete(attack_x_test, attack_feature, 1)

    # Prior distributions of attacked feature values
    priors = [(attack_x_test_feat == v).sum() / len(attack_x_test)
              for v in values]

    # white box inference attacks
    wb_attack_1 = AttributeInferenceWhiteBoxLifestyleDecisionTree(
        classifier, attack_feature=attack_feature)

    wb_attack_2 = AttributeInferenceWhiteBoxDecisionTree(
        classifier, attack_feature=attack_feature)

    # get inferred values
    inferred_train_wb1 = wb_attack_1.infer(
        attack_x_test, attack_x_test_predictions,
        values=values, priors=priors)

    inferred_train_wb2 = wb_attack_2.infer(
        attack_x_test, attack_x_test_predictions,
        values=values, priors=priors)

    # check accuracy
    actual = np.around(attack_x_test_feat, decimals=8).reshape(1, -1)
    ac1 = np.sum(inferred_train_wb1 == actual) / len(inferred_train_wb1)
    ac2 = np.sum(inferred_train_wb2 == actual) / len(inferred_train_wb2)

    print(f'White-box 1 ({label})', end=' ')
    print(f'Accuracy precision and recall:', c(f'{ac1 * 100:.2f} % '),
          c(calc_precision_recall(inferred_train_wb1, np.around(
              attack_x_test_feat, decimals=8), positive_value=0)))

    print(f'White-box 2 ({label})', end=' ')
    print(f'Accuracy precision and recall:', c(f'{ac2 * 100:.2f} % '),
          c(calc_precision_recall(inferred_train_wb2, np.around(
              attack_x_test_feat, decimals=8), positive_value=0)))


def baseline(x_train, attack_feature, label):
    """See if black-box and white-box attacks perform better than
    a baseline attack."""

    values = [0, 1]

    # training and test split
    attack_train_ratio = 0.80
    attack_train_size = int(len(x_train) * attack_train_ratio)
    attack_x_train = x_train[:attack_train_size]
    attack_x_test = x_train[attack_train_size:]

    # only attacked feature
    attack_x_test_feat = attack_x_test[:, attack_feature] \
        .copy().reshape(-1, 1)

    # training data without attacked feature
    attack_x_test = np.delete(attack_x_test, attack_feature, 1)

    baseline_attack = AttributeInferenceBaseline(
        attack_feature=attack_feature)

    # train attack model
    baseline_attack.fit(attack_x_train)

    # infer values
    inferred_train_baseline = baseline_attack.infer(
        attack_x_test, values=values)

    # check accuracy
    actual = np.around(attack_x_test_feat, decimals=8).reshape(1, -1)
    acc = np.sum(inferred_train_baseline == actual) / \
          len(inferred_train_baseline)
    print(f'Baseline attack ({label})', end=' ')
    print('Accuracy: ', c(f'{acc * 100:.2f} % '))


def inference_attack(cls, x_train, attack_feature, label):
    baseline(x_train[:], attack_feature, label)
    black_box(cls, x_train[:], attack_feature, label)
    white_box(cls, x_train[:], attack_feature, label)


def attr_inference():
    """Perform various attribute inference attacks."""

    # load decision tree model and data
    model, x_train, y_train, _, x_test, y_test = \
        train_tree(False, test_set=0.25)

    art_classifier = ScikitlearnDecisionTreeClassifier(model)
    acc = model.score(x_test, y_test)
    print('Base model accuracy: ', c(f'{acc * 100:.2f} %'))

    inference_attack(art_classifier, x_train[:], 0, 'proto=udp')
    inference_attack(art_classifier, x_train[:], 4, 'conn_state=SF')


if __name__ == '__main__':
    attr_inference()
