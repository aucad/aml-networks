"""
Apply HopSkipJumpAttack on tree-based classifier.

HopSkipJumpAttack - a family of algorithms based on a novel estimate of the
gradient direction using binary information at the decision boundary. The
proposed family includes both untargeted and targeted attacks optimized for
L2 and L∞ similarity metrics respectively. Theoretical analysis is provided
for the proposed algorithms and the gradient direction estimate.
(Chen et al., 2019) https://arxiv.org/abs/1904.02144

Loosely based on this example:
<https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/classifier_blackbox.ipynb>

Usage:

```
python src/attack_hop.py
```
"""

from sys import argv

import numpy as np
from art.attacks.evasion import HopSkipJump

import utility as tu


def run_attack(cls_loader, **cls_kwargs):
    """Run HopSkipJumpAttack attack on provided classifier

    Arguments:
         cls_loader - function to load classifier and its data
    """
    classifier, model, attrs, x, y, _, _ = cls_loader(**cls_kwargs)

    attack = HopSkipJump(
        classifier=classifier,
        targeted=False,
        max_iter=0,
        max_eval=1000,
        init_eval=10)

    max_iter = 10
    target_instance = x[0]
    iter_step = 10
    x_adv = None

    for i in range(max_iter):
        x_adv = attack.generate(
            x=np.array([target_instance]), x_adv_init=x_adv)
        print("Step %d." % (i * iter_step),
              "L2 error", np.linalg.norm(np.reshape(
                x_adv[0] - target_instance, [-1])),
              "Label %d." % np.argmax(classifier.predict(x_adv)[0]))
        attack.max_iter = iter_step


if __name__ == '__main__':
    from os import path
    from train_xg import train

    ds = argv[1] if len(argv) > 1 else tu.DEFAULT_DS
    plot_path = path.join('', 'square_result')

    run_attack(train, dataset=ds, test_size=0, max=-1, robust=False)
