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

    def __init__(self):
        super().__init__('hopskip')

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

    def attack_instance(self, attack, target, initial_label, mask):
        """Apply attack to specified instance."""

        max_iter, iter_step = 10, 10
        x_adv, success, l2_error, label = None, False, 100, initial_label
        target_arr = np.array([target])
        mask_arr = np.array([mask])

        for i in range(max_iter):
            x_adv = attack.generate(
                x=target_arr, x_adv_init=x_adv, mask=mask_arr)
            error_before = l2_error
            l2_error = np.linalg.norm(
                np.reshape(x_adv[0] - target, [-1]))
            error_change = error_before - l2_error
            label = np.argmax(self.cls.classifier.predict(x_adv)[0])
            success = initial_label != label
            attack.max_iter = iter_step
            # stop early if small error change and success
            if success and error_change < .1:
                return x_adv, success, l2_error, label

        return x_adv, success, l2_error, label

    def run(self):
        """Run HopSkipJumpAttack attack on provided classifier

        Arguments:
             cls_loader - function to load classifier and its data
             fmt - pre-prediction formatter function for single data instance
             prd - prediction function that returns class labels for given data
        """
        # create attack instance
        attack = ARTHopSkipJump(
            # a trained classifier
            classifier=self.cls.classifier,
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
        x, y = self.cls.train_x, self.cls.train_y
        ori_inputs = self.cls.formatter(x, y)
        predictions = self.cls.predict(ori_inputs).flatten().tolist()
        mask = self.get_mask(x, self.cls.mask_cols)
        self.log_attack_setup()

        # TODO: do this on the matrix, not individually
        for index, instance in enumerate(x):
            init_label = predictions[index]
            xa, success, l2, new_label = self.attack_instance(
                attack, instance, init_label, mask[index])
            ax.append(xa[:])
            ay.append(new_label)
            if success:
                evasions.append(index)
                errors.append(l2)

        self.evasions = np.array(evasions)
        self.adv_x = np.array(ax).reshape(x.shape)
        self.adv_y = np.array(ay)
        self.log_attack_stats()

        if self.attack_success:
            self.dump_result()
            self.show('Error', f'{min(errors):.6f} - {max(errors):.6f}')
