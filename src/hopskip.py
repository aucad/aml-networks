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

import numpy as np
from art.attacks.evasion import HopSkipJump as ARTHopSkipJump

from src import AbsAttack

logger = logging.getLogger(__name__)


class HopSkip(AbsAttack):

    def __init__(self, *args):
        super().__init__('hopskip', *args)
        self.max_iter = 10
        self.iter_step = 1

    @staticmethod
    def get_mask(target, freeze_indices):
        """Mask selected attributes"""

        # array with the same properties as target filled with 1s
        mask = np.full_like(target, 1)

        # set mask to 0 to prevent perturbations
        for i in range(len(target[0])):
            if i in freeze_indices:
                mask[:, i] = 0
        return mask

    def attack_instance(self, iters):
        """Create attack instance."""
        return ARTHopSkipJump(
            # a trained classifier
            classifier=self.cls.classifier,
            # size of the batch used by the estimator during inference.
            batch_size=64,
            # should the attack target one specific class.
            targeted=False,
            # Order of the norm. Possible values: "inf", np.inf or 2.
            norm=2,
            # Maximum number of iterations.
            max_iter=iters,
            # Maximum number of evaluations for estimating gradient.
            max_eval=1000,
            # Initial number of evaluations for estimating gradient.
            init_eval=100,
            # Maximum number of trials for initial generation of
            # adversarial examples.
            init_size=100,
            # Show progress bars
            verbose=True
        )

    def run(self, max_iter):
        """Run HopSkipJumpAttack attack

        Attributes:
            max_iter - max number of iterations for HopSkipJump
        """

        self.max_iter = max_iter if max_iter > 0 else self.max_iter
        x, y = self.ori_x, self.ori_y
        ori_inputs = self.cls.formatter(x, y)
        mask = self.get_mask(x, self.cls.mask_cols)
        predictions = np.array(self.cls.predict(ori_inputs)
                               .flatten().tolist())
        self.log_attack_setup()

        target = np.array(x)
        errors, labels, x_adv = None, None, None
        attack_iter = self.max_iter if self.iterated else 2
        ev_conv = 0  # count rounds where evasion # stays stable

        for mi in range(1, attack_iter, self.iter_step):
            iters = mi if self.iterated else self.max_iter
            x_adv = self.attack_instance(iters).generate(
                x_adv_init=x_adv, x=target, mask=np.array(mask))
            self.clear_one_line()
            labels = np.argmax(self.cls.classifier.predict(x_adv), 1)
            errors = np.linalg.norm((target - x_adv), axis=1)
            ev_init = len(self.evasions)
            self.evasions = np.array(
                (np.where(labels != predictions)[0]).flatten().tolist())
            ev_conv += 1 if ev_init == len(self.evasions) else 0
            if len(self.evasions) == self.n_records or ev_conv > 2:
                break

        self.adv_x = x_adv
        self.adv_y = labels
        self.validate()
        self.log_attack_stats()
        errors = errors[self.evasions] if len(self.evasions) else [0]

        if self.attack_success:
            self.plot()
            self.dump_result()
            err_min = min(errors)
            err_max = max(errors)
            self.show('L-norm', f'{err_min:.6f} - {err_max:.6f}')
