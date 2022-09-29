"""
Zeroth-Order Optimization (ZOO) evasion attack using tree-base classifier.

The black-box zeroth-order optimization attack from Pin-Yu Chen et
al. (2018). This attack is a variant of the C&W attack which uses
ADAM coordinate descent to perform numerical estimation of gradients.
Paper: https://arxiv.org/abs/1708.03999
"""

import logging
import warnings
from os import path
from itertools import combinations

import numpy as np
from art.attacks.evasion import ZooAttack
from matplotlib import pyplot as plt

import utility as tu

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")  # ignore import warnings

colors = ['deepskyblue', 'lawngreen']
diff_props = {'c': 'black', 'zorder': 2, 'lw': 1}
class_props = {'edgecolor': 'black', 'lw': .5, 's': 20, 'zorder': 2}
adv_props = {'zorder': 2, 'c': 'red', 'marker': 'x', 's': 12}

plt.rc('axes', labelsize=6)
plt.rc('xtick', labelsize=6)
plt.rc('ytick', labelsize=6)


def adversarial_iot(classifier, x_train):
    """Generate the adversarial examples."""

    return ZooAttack(
        # A trained classifier
        classifier=classifier,
        # Confidence of adversarial examples: a higher value produces
        # examples that are farther away, from the original input,
        # but classified with higher confidence as the target class.
        confidence=0.5,
        # Should the attack target one specific class
        # this doesn't matter in a binary problem!
        targeted=False,
        # The initial learning rate for the attack algorithm. Smaller
        # values produce better results but are slower to converge.
        learning_rate=1e-1,
        # The maximum number of iterations.
        max_iter=80,
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
        # True if to use the resizing strategy from the paper: first,
        # compute attack on inputs resized to 32x32, then increase
        # size if needed to 64x64, followed by 128x128.
        use_resize=False,
        # True if to use importance sampling when choosing coordinates
        # to update.
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
        .generate(x=x_train)


def adv_examples(fmt, prd, x_train, y, x_adv):
    """Make a list of adversarial instance indices that succeed in
    evasion. """
    # predictions for training instances
    ori_inputs = fmt(x_train, y) if fmt else x_train
    original = prd(ori_inputs).flatten().tolist()

    # adversarial predictions for same data
    adv_inputs = fmt(x_adv, y) if fmt else x_adv
    adversarial = prd(adv_inputs).flatten().tolist()

    # adv succeeds when predictions differ
    adv_success = [i for i, (x, y) in
                   enumerate(zip(original, adversarial))
                   if int(x) != int(y)]

    acc = 100 * len(adv_success) / len(x_adv)
    print('Zoo attack')
    tu.show('Evasion success', f'{len(adv_success)} ({acc:.2f} %)')
    return np.array(adv_success), np.array(adversarial)


def plot(img_name, evasions, attr, *data):
    """Visualize the adversarial attack results"""

    (x_train, y_train, adv_ex) = data
    class_labels = list([int(i) for i in np.unique(y_train)])

    rows, cols = 2, 3
    subplots = rows * cols
    markers = ['o', 's']
    x_min, y_min, x_max, y_max = -0.1, -0.1, 1.1, 1.1

    non_binary_attrs = tu.non_bin_attributes(x_train)
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
                ax.plot([xt_f1, ad_f1], [xt_f2, ad_f2], **diff_props)
                ax.scatter(ad_f1, ad_f2, **adv_props)

        fig.tight_layout()
        tu.ensure_out_dir(tu.RESULT_DIR)
        plt.savefig(path.join(tu.RESULT_DIR, f'{img_name}_{f + 1}.png'))


def post_restore(mask_idx, x, adv):
    """Restore attribute values along immutable columns."""
    if x.shape != adv.shape:
        raise Exception(f'Shapes do not match {x.shape} {adv.shape}')
    for idx in mask_idx:
        ori_values = x[:, idx]
        adv[:, idx] = ori_values
    return adv


def zoo_attack(cls):
    """Carry out ZOO attack on specified classifier.

    Arguments:
         cls_loader - function to load classifier and its data
         fmt - pre-prediction formatter function for a data instance
         prd - prediction function that returns class labels for data
         img_path - dir path and file name for storing plots
    """
    classifier = cls.classifier
    attrs = cls.attrs
    x = cls.train_x
    y = cls.train_y
    fmt = cls.formatter
    prd = cls.predict

    int_attrs = tu.freeze_types(x)
    tu.show('Immutable', ", ".join([attrs[i] for i in int_attrs]))
    tu.show('Mutable', ", ".join(sorted(
        [attrs[i] for i in range(len(attrs) - 1) if
         i not in int_attrs])))

    x_train, labels = x[:], y[:]
    adv = adversarial_iot(classifier, x)
    adv = post_restore(int_attrs, x_train, adv)
    data = (x_train, labels, adv)
    evasions, adv_y = adv_examples(fmt, prd, *data)

    if len(evasions) > 0:
        plot(tu.RESULT_DIR, evasions, attrs, *data)
        tu.dump_result(evasions, x_train, labels, adv, adv_y, attrs)

    return evasions


if __name__ == '__main__':
    from loader import ClsLoader

    cls = ClsLoader.load(ClsLoader.XGBOOST)\
        .load(tu.DEFAULT_DS, .95).train()
    zoo_attack(cls)
