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
from os import path
from itertools import combinations

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
    model, x_train, y_train, ATTR = train_tree(False)

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

    print('Train scores:', c("%.4f" % ori_score))
    print('Adv scores:  ', c("%.4f" % adv_score))
    print('# Evasions:  ', c(len(adv_success)))
    return np.array(adv_success)


def plot(model, class_labels, evasions, attr, *data):
    """Visualize the adversarial attack results"""

    (x_train, y_train, x_train_adv) = data

    # range for contour plot
    levels, h = [x / 10. for x in range(0, 11)], .1
    colors = ['deepskyblue', 'lawngreen']

    avg_sample = np.mean(x_train, axis=0)

    non_bin_attrs = []
    for feat in range(len(x_train[0])):
        if len(list(set(x_train[:, feat]))) > 2:
            non_bin_attrs.append(feat)
    attr_pairs = list(combinations(non_bin_attrs, 2))

    rows, cols = 2, 3
    fig_count = len(attr_pairs) // (rows * cols)

    # generate plots
    for f in range(fig_count):
        fig, axs = plt.subplots(
            figsize=[7., 5],
            nrows=rows, ncols=cols, dpi=250,
            constrained_layout=True)
        axs = axs.flatten()

        for p in range(rows * cols):
            f1, f2 = attr_pairs[(f * rows * cols) + p]
            x_min, x_max = -0.1, 1.1
            y_min, y_max = x_min, x_max

            axs[p].set_xlim((x_min, x_max))
            axs[p].set_ylim((y_min, y_max))
            axs[p].set_xlabel(attr[f1])
            axs[p].set_ylabel(attr[f2])
            axs[p].set_aspect('equal', adjustable='box')

            for j in evasions:
                # Plot difference vectors
                axs[p].plot(
                    [x_train[j:j + 1, f1], x_train_adv[j:j + 1, f1]],
                    [x_train[j:j + 1, f2], x_train_adv[j:j + 1, f2]],
                    c='black', zorder=2, lw=1)

            # Plot original samples
            for cl in range(len(class_labels)):
                axs[p].scatter(
                    x_train[y_train == cl][:, f1],
                    x_train[y_train == cl][:, f2],
                    edgecolor='black', linewidth=.5,
                    s=20, zorder=2, c=colors[cl])

            for j in evasions:
                # Plot adversarial examples
                axs[p].scatter(
                    x_train_adv[j:j + 1, f1],
                    x_train_adv[j:j + 1, f2],
                    zorder=2, c='red', marker='x', s=20)

            # Show predicted probability as contour plot
            xx, yy = np.meshgrid(
                np.arange(x_min, x_max, h),
                np.arange(y_min, y_max, h))

            pred_array = np.tile(avg_sample, (len(xx.ravel()), 1))
            pred_array[:, f1] = xx.ravel()
            pred_array[:, f2] = yy.ravel()

            # plot class probabilities as contour plot
            for i, class_label in enumerate(class_labels):
                z_prob = model.predict_proba(pred_array)[:, i]
                im = axs[p].contourf(
                    xx, yy, z_prob.reshape(xx.shape),
                    levels=levels[:], vmin=0, vmax=1)
                # if i == len(class_labels) - 1:
                # cax = fig.add_axes([0.1, 0.04, 0.33, 0.01])
                # plt.colorbar(im, ax=axs[i], cax=cax, ticks=[0, 1],
                #              orientation='horizontal')

        fig.tight_layout(pad=.25)
        # fig.subplots_adjust(bottom=.08, right=.97, hspace=.05)
        plt.savefig(f'{IMAGE_NAME}_{f + 1}.png', pad_inces=.5)


if __name__ == '__main__':
    cls, data, attr = adversarial_iot()
    evasions = adv_examples(cls, *data)
    labels = list(set(data[1]))
    plot(cls, labels, evasions, attr, *data)
