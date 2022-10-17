"""
Zeroth-Order Optimization (ZOO) evasion attack using tree-base classifier.

The black-box zeroth-order optimization attack from Pin-Yu Chen et
al. (2018). This attack is a variant of the C&W attack which uses
ADAM coordinate descent to perform numerical estimation of gradients.
Paper: https://arxiv.org/abs/1708.03999
"""

from itertools import combinations

import numpy as np
from art.attacks.evasion import ZooAttack as ARTZooAttack
from matplotlib import pyplot as plt

from src import AbsAttack


class Zoo(AbsAttack):

    def __init__(self, iterated):
        super().__init__('zoo', iterated)
        self.max_iter = 80
        self.iter_step = 10

    def generate_adv_examples(self, iters) -> None:
        """Generate the adversarial examples using ZOO attack."""

        self.adv_x = ARTZooAttack(
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
            verbose=True) \
            .generate(x=self.cls.train_x)

    @staticmethod
    def non_bin_attributes(np_array):
        """Get column indices of non-binary attributes"""
        return [feat for feat in range(len(np_array[0]))
                if len(list(set(np_array[:, feat]))) > 2]

    @staticmethod
    def pseudo_mask(mask_idx, x, adv):
        """
        Restore original attribute values along immutable columns.

        We perform this operation after the attack, because masking
        is not supported for this attack.
        """
        if x.shape != adv.shape:
            raise Exception(
                f'Shapes do not match {x.shape} {adv.shape}')
        for idx in mask_idx:
            ori_values = x[:, idx]
            adv[:, idx] = ori_values
        return adv

    def eval_adv_examples(self):
        """
        Do prediction on adversarial vs original records and determine
        which records succeed at evading expected class label.
        """
        # predictions for training instances
        ori_in = self.cls.formatter(self.cls.train_x, self.cls.train_y)
        original = self.cls.predict(ori_in).flatten().tolist()

        # adversarial predictions for same data
        adv_in = self.cls.formatter(self.adv_x, self.cls.train_y)
        adversarial = self.cls.predict(adv_in).flatten().tolist()
        self.adv_y = np.array(adversarial)

        # adversary succeeds iff predictions differ
        self.evasions = np.array(
            [i for i, (x, y) in enumerate(zip(original, adversarial))
             if int(x) != int(y)])

    def run(self, max_iter):
        """Runs the zoo attack."""

        self.max_iter = max_iter if max_iter > 0 else self.max_iter
        attack_iter = self.max_iter if self.iterated else 1
        self.log_attack_setup()

        for mi in range(0, attack_iter, self.iter_step):
            iters = mi if self.iterated else self.max_iter
            self.generate_adv_examples(iters)
            self.adv_x = self.pseudo_mask(
                self.cls.mask_cols, self.cls.train_x, self.adv_x)
            self.eval_adv_examples()
            if len(self.evasions == self.cls.n_train):
                break

        self.validate()
        self.log_attack_stats()

        if self.attack_success:
            self.plot()
            self.dump_result()

    def plot(self):
        """Visualize the adversarial attack results"""

        colors = ['deepskyblue', 'lawngreen']
        diff_props = {'c': 'black', 'zorder': 2, 'lw': 1}
        class_props = {'edgecolor': 'black', 'lw': .5, 's': 20,
                       'zorder': 2}
        adv_props = {'zorder': 2, 'c': 'red', 'marker': 'x', 's': 12}
        plt.rc('axes', labelsize=6)
        plt.rc('xtick', labelsize=6)
        plt.rc('ytick', labelsize=6)

        evasions, attr, adv_ex = \
            self.evasions, self.cls.attrs, self.adv_x
        x_train, y_train = self.cls.train_x, self.cls.train_y
        class_labels = list([int(i) for i in np.unique(y_train)])

        rows, cols = 2, 3
        subplots = rows * cols
        markers = ['o', 's']
        x_min, y_min, x_max, y_max = -0.1, -0.1, 1.1, 1.1

        non_binary_attrs = self.non_bin_attributes(x_train)
        attr_pairs = list(combinations(non_binary_attrs, 2))
        fig_count = len(attr_pairs) // subplots

        # generate plots
        for f in range(fig_count):
            fig, axs = plt.subplots(nrows=rows, ncols=cols, dpi=250)
            axs = axs.flatten()

            for subplot_index in range(subplots):

                ax = axs[subplot_index]
                f1, f2 = attr_pairs[(f * subplots) + subplot_index]
                ax.set_xlim((x_min, x_max))
                ax.set_ylim((y_min, y_max))
                ax.set_xlabel(attr[f1])
                ax.set_ylabel(attr[f2])
                ax.set_aspect('equal', adjustable='box')

                # Plot original samples
                for cl in class_labels:
                    x_f1 = x_train[y_train == cl][:, f1]
                    x_f2 = x_train[y_train == cl][:, f2]
                    style = {'c': colors[cl], 'marker': markers[cl]}
                    ax.scatter(x_f1, x_f2, **class_props, **style)

                # Plot adversarial examples and difference vectors
                for j in evasions:
                    xt_f1 = x_train[j:j + 1, f1]
                    xt_f2 = x_train[j:j + 1, f2]
                    ad_f1 = adv_ex[j:j + 1, f1]
                    ad_f2 = adv_ex[j:j + 1, f2]
                    ax.plot([xt_f1, ad_f1], [xt_f2, ad_f2],
                            **diff_props)
                    ax.scatter(ad_f1, ad_f2, **adv_props)

            fig.tight_layout()
            self.ensure_out_dir(self.out_dir)
            plt.savefig(self.figure_name(f + 1))
