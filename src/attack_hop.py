"""
Applying HopSkipJumpAttack on tree-based classifier.

HopSkipJumpAttack - a family of algorithms based on a novel estimate
of the gradient direction using binary information at the decision
boundary. The proposed family includes both untargeted and targeted
attacks optimized for L2 and L∞ similarity metrics respectively.
(Chen et al., 2019) Paper: https://arxiv.org/abs/1904.02144

Implementation lLoosely based on this example:
<https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/classifier_blackbox.ipynb>

Usage:

```
python src/attack_hop.py
```
"""

from sys import argv
from math import fabs

import numpy as np
from art.attacks.evasion import HopSkipJump

import utility as tu


def attack_instance(classifier, attack, target, initial_label):
    """Apply attack to specified instance."""

    # TODO: adjust this to attack a vector instead of single
    #   instance

    max_iter, iter_step = 10, 10
    x_adv, success, l2_error, label = None, False, 100, initial_label

    for i in range(max_iter):
        # TODO: mask selected attributes
        x_adv = attack.generate(x=np.array([target]), x_adv_init=x_adv)
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


def run_attack(cls_loader, fmt, prd, **cls_kwargs):
    """Run HopSkipJumpAttack attack on provided classifier

    Arguments:
         cls_loader - function to load classifier and its data
         fmt - pre-prediction formatter function for single data instance
         prd - prediction function that returns class labels for given data
    """
    classifier, model, attrs, x, y, _, _ = cls_loader(**cls_kwargs)

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

    x_adv, errors = [], []
    ori_inputs = fmt(x, y) if fmt else x
    predictions = prd(model, ori_inputs).flatten().tolist()
    mutations = set()

    for index, instance in enumerate(x):
        init_label = predictions[index]
        xa, success, l2, new_label = attack_instance(
            classifier, attack, instance, init_label)
        if success:
            x_adv.append([(xa, new_label), (instance, init_label)])
            errors.append(l2)
            for i, attr_o in enumerate(instance):
                attr_a = xa[0, i]
                diff = fabs(attr_o - attr_a)
                if diff > 0.01:
                    mutations.add(attrs[i])

    print('-' * 50)
    print('HOPSKIPJUMP ATTACK')
    print('-' * 50)
    tu.show('Success rate', f'{(len(x_adv) / len(x)) * 100:.2f}')
    tu.show(f'Error min/max', f'{min(errors):.6f} - {max(errors):.6f}')
    tu.show(f'mutations:', f'{len(list(mutations))} attributes')
    print(" :: ".join(list(mutations)))

    # TODO: plot these results (similar to zoo attack)


if __name__ == '__main__':
    from train_xg import train, formatter, predict

    ds = argv[1] if len(argv) > 1 else tu.DEFAULT_DS
    run_attack(
        train, formatter, predict,
        dataset=ds, test_size=0, max=-1, robust=True)
