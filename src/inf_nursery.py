"""
Simple adversarial example using ART with scikit-learn and applying
attribute inference attack.

Inspired by this example:

https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/
084e9769fc84934f77ec600ced5452a0e9baa60f/notebooks/
attack_attribute_inference.ipynb

Usage:

```
python src/inf_nursery.py
```
"""

import warnings

warnings.filterwarnings("ignore")  # ignore import warnings

import numpy as np
from art.attacks.inference.attribute_inference \
    import AttributeInferenceBlackBox
from art.attacks.inference.attribute_inference \
    import AttributeInferenceWhiteBoxLifestyleDecisionTree
from art.attacks.inference.attribute_inference \
    import AttributeInferenceWhiteBoxDecisionTree
from art.attacks.inference.attribute_inference \
    import AttributeInferenceBaseline
# from art.attacks.inference.membership_inference \
#     import MembershipInferenceBlackBox
# from art.attacks.inference.attribute_inference \
#     import AttributeInferenceMembership
from art.estimators.classification.scikitlearn \
    import ScikitlearnDecisionTreeClassifier
from art.utils import load_nursery
from sklearn.tree import DecisionTreeClassifier


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

    return precision, recall


def nursery_demo():
    """Replicating example from notebook."""

    # 1. load data
    (x_train, y_train), (x_test, y_test), _, _ = load_nursery(
        test_set=0.2, transform_social=True)

    # 2. train the model
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    art_classifier = ScikitlearnDecisionTreeClassifier(model)
    print('Base model accuracy: ', model.score(x_test, y_test))

    # (I) Black-box attack
    # --------------------
    # The black-box attack basically trains an
    # additional classifier (called the attack model) to predict the
    # attacked feature's value from the remaining n-1 features as
    # well as the original (attacked) model's predictions.

    # Train attack model
    attack_train_ratio = 0.5
    attack_train_size = int(len(x_train) * attack_train_ratio)
    attack_x_train = x_train[:attack_train_size]
    attack_y_train = y_train[:attack_train_size]
    attack_x_test = x_train[attack_train_size:]
    attack_y_test = y_train[attack_train_size:]

    attack_feature = 1  # social

    # get original model's predictions
    attack_x_test_predictions = np.array(
        [np.argmax(arr) for arr in
         art_classifier.predict(attack_x_test)]).reshape(-1, 1)

    # only attacked feature
    attack_x_test_feature = \
        attack_x_test[:, attack_feature].copy().reshape(-1, 1)

    # training data without attacked feature
    attack_x_test = np.delete(attack_x_test, attack_feature, 1)

    bb_attack = AttributeInferenceBlackBox(
        art_classifier, attack_feature=attack_feature)

    # train attack model
    bb_attack.fit(attack_x_train)

    # Infer sensitive feature and check accuracy
    # get inferred values
    values = [-0.70718864, 1.41404987]
    inferred_train_bb = bb_attack.infer(
        attack_x_test, pred=attack_x_test_predictions, values=values)

    # check accuracy
    train_acc = np.sum(inferred_train_bb == np.around(
        attack_x_test_feature, decimals=8).reshape(1, -1)) / len(
        inferred_train_bb)
    print(f'Blackbox accuracy: {train_acc}')

    # This means that for {train_acc} of the training set, the
    # attacked feature is inferred correctly using this attack.

    # (II) Whitebox attacks
    # ---------------------
    # These two attacks do not train any
    # additional model, they simply use additional information coded
    # within the attacked decision tree model to compute the
    # probability of each value of the attacked feature and outputs
    # the value with the highest probability.

    # First attack

    wb_attack = AttributeInferenceWhiteBoxLifestyleDecisionTree(
        art_classifier, attack_feature=attack_feature)

    priors = [3465 / 5183, 1718 / 5183]

    # get inferred values
    inferred_train_wb1 = wb_attack.infer(
        attack_x_test, attack_x_test_predictions,
        values=values, priors=priors)

    # check accuracy
    train_acc = np.sum(inferred_train_wb1 == np.around(
        attack_x_test_feature, decimals=8).reshape(1, -1)) / len(
        inferred_train_wb1)
    print(f'First whitebox accuracy: {train_acc}')  # N

    # Second attack

    wb2_attack = AttributeInferenceWhiteBoxDecisionTree(
        art_classifier, attack_feature=attack_feature)

    # get inferred values
    inferred_train_wb2 = wb2_attack.infer(
        attack_x_test, attack_x_test_predictions,
        values=values, priors=priors)

    # check accuracy
    train_acc = np.sum(inferred_train_wb2 == np.around(
        attack_x_test_feature, decimals=8).reshape(1, -1)) / len(
        inferred_train_wb2)
    print(f'Second whitebox accuracy: {train_acc}')  # M

    # The white-box attacks are able to correctly infer the attacked
    # feature value in {N}% and {M}% of the training set respectively.

    # PRECISION AND RECALL

    # black-box
    print('black box precision and recall', calc_precision_recall(
        inferred_train_bb,
        np.around(attack_x_test_feature, decimals=8),
        positive_value=1.41404987))

    # white-box 1
    print('white box I: precision and recall', calc_precision_recall(
        inferred_train_wb1,
        np.around(attack_x_test_feature, decimals=8),
        positive_value=1.41404987))

    # white-box 2
    print('white box II: precision and recall', calc_precision_recall(
        inferred_train_wb2,
        np.around(attack_x_test_feature, decimals=8),
        positive_value=1.41404987))

    # To verify the significance of these results, we now run a
    # baseline attack that uses only the remaining features to try to
    # predict the value of the attacked feature, with no use of the
    # model itself.

    baseline_attack = AttributeInferenceBaseline(
        attack_feature=attack_feature)

    # train attack model
    baseline_attack.fit(attack_x_train)
    # infer values
    inferred_train_baseline = baseline_attack.infer(
        attack_x_test, values=values)
    # check accuracy
    baseline_train_acc = np.sum(
        inferred_train_baseline == np.around(
            attack_x_test_feature, decimals=8)
        .reshape(1, -1)) / len(inferred_train_baseline)
    print('baseline accuracy: ', baseline_train_acc)

    # Should now see that both the black-box and white-box attacks do
    # better than the baseline.

    # (III) Membership based attack
    # -----------------------------
    #
    # <<<<<< THIS NEEDS PYTORCH >>>>>>>
    #
    # In this attack the idea is to find the target feature value that
    # maximizes the membership attack confidence, indicating that
    # this is the most probable value for member samples. It can be
    # based on any membership attack (either black-box or white-box)
    # as long as it supports the given model.

    # train membership attack
    # mem_attack = MembershipInferenceBlackBox(art_classifier)
    # mem_attack.fit(x_train[:attack_train_size],
    #                y_train[:attack_train_size], x_test, y_test)
    #
    # # Apply attribute attack
    # attack = AttributeInferenceMembership(
    #     art_classifier, mem_attack, attack_feature=attack_feature)
    #
    # # infer values
    # inferred_train = attack.infer(
    #     attack_x_test, attack_y_test, values=values)
    #
    # # check accuracy
    # train_acc = np.sum(inferred_train == np.around(
    #     attack_x_test_feature, decimals=8).reshape(1, -1)) / len(
    #     inferred_train)
    #
    # print('membership (blackbox) accuracy', train_acc)

    # Should see that this attack does slightly better than the
    # regular black-box attack, even though it still assumes only
    # black-box access to the model (employs a black-box membership
    # attack). But it is not as good as the white-box attacks.


if __name__ == '__main__':
    nursery_demo()
