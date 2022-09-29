"""
Applying HopSkipJumpAttack on tree-based classifier.

HopSkipJumpAttack - a family of algorithms based on a novel estimate
of the gradient direction using binary information at the decision
boundary. The proposed family includes both untargeted and targeted
attacks optimized for L2 and L∞ similarity metrics respectively.
(Chen et al., 2019) Paper: https://arxiv.org/abs/1904.02144

Implementation loosely based on this example:
<https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/classifier_blackbox.ipynb>
"""

import logging
from math import fabs

import numpy as np
from art.attacks.evasion import HopSkipJump

import utility as tu

logger = logging.getLogger(__name__)


def get_mask(target, freeze_indices):
    """Mask selected attributes"""

    # array with the same properties as target filled with 1s
    mask = np.full_like(target, 1)

    # set mask to 0 to prevent perturbations
    for i in range(len(target[0])):
        if i in freeze_indices:
            mask[:, i] = 0

    return mask


def attack_instance(
        classifier, attack, target, initial_label, mask=None
):
    """Apply attack to specified instance."""

    max_iter, iter_step = 10, 10
    x_adv, success, l2_error, label = None, False, 100, initial_label
    target_arr = np.array([target])
    mask_arr = np.array([mask])

    for i in range(max_iter):
        x_adv = attack.generate(
            x=target_arr, x_adv_init=x_adv, mask=mask_arr)
        error_before = l2_error
        l2_error = np.linalg.norm(np.reshape(x_adv[0] - target, [-1]))
        error_change = error_before - l2_error
        label = np.argmax(classifier.predict(x_adv)[0])
        success = initial_label != label
        attack.max_iter = iter_step
        # stop early if small error change and success
        if success and error_change < .1:
            return x_adv, success, l2_error, label

    return x_adv, success, l2_error, label


def run_attack(cls):
    """Run HopSkipJumpAttack attack on provided classifier

    Arguments:
         cls_loader - function to load classifier and its data
         fmt - pre-prediction formatter function for single data instance
         prd - prediction function that returns class labels for given data
    """
    classifier = cls.classifier
    attrs = cls.attrs
    x = cls.train_x
    y = cls.train_y
    fmt = cls.formatter
    prd = cls.predict

    # create attack instance
    attack = HopSkipJump(
        # a trained classifier
        classifier=classifier,
        # size of the batch used by the estimator during inference.
        batch_size=64,
        # should the attack target one specific class.
        targeted=False,
        # Order of the norm. Possible values: "inf", np.inf or 2.
        norm=2,
        # Maximum number of iterations.
        max_iter=0,
        # Maximum number of evaluations for estimating gradient.
        max_eval=1000,
        # Initial number of evaluations for estimating gradient.
        init_eval=100,
        # Maximum number of trials for initial generation of
        # adversarial examples.
        init_size=100,
        # Show progress bars
        verbose=False
    )

    ax, ay = [], []
    x_adv, errors, evasions = [], [], []
    ori_inputs = fmt(x, y) if fmt else x
    predictions = prd(ori_inputs).flatten().tolist()
    mutations = set()
    int_attrs = tu.freeze_types(x)
    mask = get_mask(x, int_attrs)

    print('HopSkipJump')
    tu.show('Immutable', ", ".join([attrs[i] for i in int_attrs]))
    tu.show('Mutable', ", ".join(sorted(
        [attrs[i] for i in range(len(attrs) - 1) if
         i not in int_attrs])))

    for index, instance in enumerate(x):
        init_label = predictions[index]
        xa, success, l2, new_label = attack_instance(
            classifier, attack, instance, init_label, mask[index])
        ax.append(xa[:])
        ay.append(new_label)
        if success:
            evasions.append(index)
            errors.append(l2)
            for i, attr_o in enumerate(instance):
                if fabs(attr_o - xa[0, i]) > 0.0001:
                    mutations.add(attrs[i])
    evs, mut = len(evasions), list(mutations)
    if evs > 0:
        ax = np.array(ax).reshape(x.shape)
        ay, evasions = np.array(ay), np.array(evasions)
        tu.dump_result(evasions, x, y, ax, ay, attrs)

    tu.show('Evasion success', f'{evs} / {(evs / len(x)) * 100:.2f} %')
    if evs > 0:
        tu.show('Error', f'{min(errors):.6f} - {max(errors):.6f}')
    if len(mut) > 0:
        tu.show('Mutations:', f'{len(mut)} attributes')
        tu.show('Mutated attrs', ", ".join(sorted(mut)))
    return evs


if __name__ == '__main__':
    from loader import ClsLoader

    cls = ClsLoader.load(ClsLoader.XGBOOST)\
        .load(tu.DEFAULT_DS, .95).train()
    run_attack(cls)
