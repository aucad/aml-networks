"""
Applying HopSkipJumpAttack on tree-based classifier.

HopSkipJumpAttack - a family of algorithms based on a novel estimate
of the gradient direction using binary information at the decision
boundary. The proposed family includes both un-targeted and targeted
attacks optimized for L2 and L∞ similarity metrics respectively.
(Chen et al., 2019) Paper: https://arxiv.org/abs/1904.02144
"""
from typing import List

import numpy as np
from art.attacks.evasion import HopSkipJump as ARTHopSkipJump

from src import Attack, utility


class HopSkip(Attack):

    def __init__(self, *args):
        super().__init__('hopskipjump', 10, *args)

    @staticmethod
    def get_mask(target: np.array, freeze: List[int]) -> np.array:
        """Mask selected attributes"""

        # array with the same properties as target filled with 1s
        mask = np.full_like(target, 1)

        # set mask to 0 to prevent perturbations
        for i in range(len(target[0])):
            if i in freeze:
                mask[:, i] = 0
        return mask

    def generate_adv_examples(
            self, iters: int, x: np.array, mask: np.array):
        """Create attack instance."""
        self.adv_x = ARTHopSkipJump(
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
            verbose=not self.silent
        ).generate(x_adv_init=None, x=x, mask=np.array(mask))
        if not self.silent:
            utility.clear_one_line()

    def run(self):
        """Run HopSkipJumpAttack attack."""
        mask = self.get_mask(self.ori_x, self.cls.mask_cols)
        self.generate_adv_examples(self.max_iter, self.ori_x, mask)
        self.eval_examples()
        self.post_run()
