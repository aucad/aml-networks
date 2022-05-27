"""
Simple adversarial example using ART with scikit-learn and applying
Zeroth-Order Optimization (ZOO) Attack.

The black-box zeroth-order optimization attack from Pin-Yu Chen et
al. (2018). This attack is a variant of the C&W attack which uses
ADAM coordinate descent to perform numerical estimation of gradients.

Usage:

```
python src/adv_iot.py
```
"""

import warnings
from itertools import combinations
from os import path

warnings.filterwarnings("ignore")  # ignore import warnings

import numpy as np
from art.attacks.evasion import ZooAttack
from art.estimators.classification import SklearnClassifier
from matplotlib import pyplot as plt

from tree import train_tree, text_label
from utility import color_text as c

IMAGE_NAME = path.join('adversarial', 'iot-23')

plt.rc('axes', labelsize=6)
plt.rc('xtick', labelsize=6)
plt.rc('ytick', labelsize=6)


def adversarial_iot():
    """Generate the adversarial examples."""
    # get the tree model and its training data
    model, x_train, y_train, ATTR, _, _ = train_tree(False)

    # Create ART Zeroth Order Optimization attack
    # using scikit-learn DecisionTreeClassifier
    zoo = ZooAttack(
        # A trained classifier
        classifier=SklearnClassifier(model=model),
        # Confidence of adversarial examples: a higher value produces
        # examples that are farther away, from the original input,
        # but classified with higher confidence as the target class.
        confidence=0.75,
        # Should the attack target one specific class
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
        # Number of coordinate updates to run in parallel. A higher
        # value for nb_parallel should be preferred over a large
        # batch size.
        nb_parallel=2,
        # Internal size of batches on which adversarial samples are
        # generated. Small batch sizes are encouraged for ZOO,
        # as the algorithm already runs nb_parallel coordinate
        # updates in parallel for each sample. The batch size is a
        # multiplier of nb_parallel in terms of memory consumption.
        batch_size=1,
        # Step size for numerical estimation of derivatives.
        variable_h=0.2,
        # Show progress bars.
        verbose=True)

    # train adversarial examples
    x_train_adv = zoo.generate(x_train)

    return model, (x_train, y_train, x_train_adv), ATTR


def adv_examples(model, x_train, y_train, x_train_adv):
    ori_score = model.score(x_train, y_train),
    adv_score = model.score(x_train, y_train)
    adv_success = []

    for i in range(len(x_train_adv)):
        o, a = x_train[i:i + 1, :], x_train_adv[i:i + 1, :]
        ori_pred = text_label(model.predict(o)[0])
        adv_pred = text_label(model.predict(a)[0])
        if ori_pred != adv_pred:
            adv_success.append(i)

    print('Train scores '.ljust(20, '-'), c("%.4f" % ori_score))
    print('Adv scores '.ljust(20, '-'), c("%.4f" % adv_score))
    print('# Evasions '.ljust(20, '-'), c(len(adv_success)))
    return np.array(adv_success)


def plot(evasions, attr, *data):
    """Visualize the adversarial attack results"""

    (x_train, y_train, x_train_adv) = data
    class_labels = list(set(y_train))

    colors = ['deepskyblue', 'lawngreen']
    markers = ['o', 's']
    rows, cols = 2, 3
    num_sub_plots = rows * cols
    diff_props = {'c': 'black', 'zorder': 2, 'lw': 1}
    class_props = {'edgecolor': 'black', 'lw': .5, 's': 20, 'zorder': 2}
    adv_props = {'zorder': 2, 'c': 'red', 'marker': 'x', 's': 15}
    x_min, y_min, x_max, y_max = -0.1, -0.1, 1.1, 1.1

    non_binary_attrs = [feat for feat in range(len(x_train[0])) if
                        len(list(set(x_train[:, feat]))) > 2]
    attr_pairs = list(combinations(non_binary_attrs, 2))
    fig_count = len(attr_pairs) // num_sub_plots

    # generate plots
    for f in range(fig_count):
        fig, axs = plt.subplots(nrows=rows, ncols=cols, dpi=250)
        axs = axs.flatten()

        for subplot_index in range(num_sub_plots):

            ax = axs[subplot_index]
            f1, f2 = attr_pairs[(f * num_sub_plots) + subplot_index]
            ax.set_xlim((x_min, x_max))
            ax.set_ylim((y_min, y_max))
            ax.set_xlabel(attr[f1])
            ax.set_ylabel(attr[f2])
            ax.set_aspect('equal', adjustable='box')

            # Plot original samples
            for cl in range(len(class_labels)):
                x_f1 = x_train[y_train == cl][:, f1]
                x_f2 = x_train[y_train == cl][:, f2]
                style = {'c': colors[cl], 'marker': markers[cl]}
                ax.scatter(x_f1, x_f2, **class_props, **style)

            # Plot adversarial examples and difference vectors
            for j in evasions:
                xt_f1 = x_train[j:j + 1, f1]
                xt_f2 = x_train[j:j + 1, f2]
                ad_f1 = x_train_adv[j:j + 1, f1]
                ad_f2 = x_train_adv[j:j + 1, f2]
                ax.plot([xt_f1, ad_f1], [xt_f2, ad_f2], **diff_props)
                ax.scatter(ad_f1, ad_f2, **adv_props)

        fig.tight_layout()
        plt.savefig(f'{IMAGE_NAME}_{f + 1}.png')


if __name__ == '__main__':
    cls, data, attr = adversarial_iot()
    evasions = adv_examples(cls, *data)
    plot(evasions, attr, *data)
