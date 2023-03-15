"""
Implementation of Zeroth-Order Optimization (ZOO) evasion attack.

The black-box zeroth-order optimization attack from Pin-Yu Chen et
al. (2018). This attack is a variant of the C&W attack which uses
ADAM coordinate descent to perform numerical estimation of gradients.

Paper: https://arxiv.org/abs/1708.03999
"""

from art.attacks.evasion import ZooAttack

from src import Attack, utility


class Zoo(Attack):

    def __init__(self, *args):
        super().__init__('zoo', 80, *args)

    def generate_adv_examples(self, iters: int) -> None:
        """Generate the adversarial examples using ZOO attack.

        Arguments:
            iters - number of maximum iterations
        """
        self.adv_x = ZooAttack(
            # A trained classifier
            classifier=self.cls.classifier,
            # Confidence of adversarial examples: a higher value
            # produces examples that are farther away, from the
            # original input, but classified with higher confidence
            # as the target class.
            confidence=0.5,
            # Should the attack target one specific class
            # this doesn't matter in a binary problem!
            targeted=False,
            # The initial learning rate for the attack algorithm.
            # Smaller values produce better results but are slower to
            # converge.
            learning_rate=1e-1,
            # The maximum number of iterations.
            max_iter=iters,
            # Number of times to adjust constant with binary search
            # (positive value).
            binary_search_steps=10,
            # The initial trade-off constant c to use to tune the
            # relative importance of distance and confidence. If
            # binary_search_steps is large, the initial constant is not
            # important, as discussed in Carlini and Wagner (2016).
            initial_const=1e-3,
            # True if gradient descent should be abandoned when it gets
            # stuck.
            abort_early=True,
            # True if to use the resizing strategy from the paper:
            # first, compute attack on inputs resized to 32x32,
            # then increase size if needed to 64x64, followed by
            # 128x128.
            use_resize=False,
            # True if to use importance sampling when choosing
            # coordinates to update.
            use_importance=False,
            # Number of coordinate updates to run in parallel.
            nb_parallel=5,
            # Internal size of batches on which adversarial samples are
            # generated. Only size 1 is supported.
            batch_size=1,
            # Step size for numerical estimation of derivatives.
            variable_h=0.8,
            # Show progress bar.
            verbose=not self.silent) \
            .generate(x=self.ori_x)
        if not self.silent:
            utility.clear_one_line()

    @staticmethod
    def pseudo_mask(mask_idx, ori, adv):
        """
        Restore original attribute values along immutable columns.

        We perform this operation after the attack, because masking
        is not supported by ZOO attack natively.

        Arguments:
            mask_idx - masked indices, List[int]
            ori - original records
            adv - adversarial records

        Returns:
            Modified adversarial records, with pseudo masking applied.
        """
        if ori.shape != adv.shape:
            raise Exception(
                f'Shapes do not match {ori.shape} {adv.shape}')
        for idx in mask_idx:
            ori_values = ori[:, idx]
            adv[:, idx] = ori_values
        return adv

    def run(self) -> None:
        """Run the zoo attack."""
        self.generate_adv_examples(self.max_iter)
        self.adv_x = self.pseudo_mask(
            self.cls.mask_cols, self.ori_x, self.adv_x)
        self.eval_examples()
        self.post_run()
