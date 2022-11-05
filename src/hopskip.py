"""
Applying HopSkipJumpAttack on tree-based classifier.

HopSkipJumpAttack - a family of algorithms based on a novel estimate
of the gradient direction using binary information at the decision
boundary. The proposed family includes both un-targeted and targeted
attacks optimized for L2 and L∞ similarity metrics respectively.
(Chen et al., 2019) Paper: https://arxiv.org/abs/1904.02144
"""

import numpy as np
from art.attacks.evasion import HopSkipJump as ARTHopSkipJump

from src import Attack, utility


class HopSkip(Attack):

    def __init__(self, *args):
        super().__init__('hop', 10, 1, *args)

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

    def run(self):
        """Run HopSkipJumpAttack attack."""
        x, y = self.ori_x, self.ori_y
        ori_inputs = self.cls.formatter(x, y)
        mask = self.get_mask(x, self.cls.mask_cols)
        predictions = np.array(self.cls.predict(ori_inputs)
                               .flatten().tolist())
        correct = np.array((np.where(np.array(y) == predictions)[0])
                           .flatten().tolist())
        target = np.array(x)
        labels, x_adv = None, None
        attack_iter = self.max_iter if self.iterated else 2
        ev_conv = 0  # count rounds where evasion # stays stable

        for mi in range(1, attack_iter, self.iter_step):
            iters = mi if self.iterated else self.max_iter
            x_adv = self.attack_instance(iters).generate(
                x_adv_init=x_adv, x=target, mask=np.array(mask))
            utility.clear_one_line()
            labels = np.argmax(self.cls.classifier.predict(x_adv), 1)
            ev_init = len(self.evasions)
            evades = np.array((np.where(labels != predictions)[0])
                              .flatten().tolist())
            self.evasions = np.intersect1d(evades, correct)
            ev_conv += 1 if ev_init == len(self.evasions) else 0
            if self.n_evasions == self.n_records or ev_conv > 2:
                break

        self.adv_x = x_adv
        self.adv_y = labels
        self.post_run()
