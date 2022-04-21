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
from os import path

import numpy as np
from art.attacks.evasion import ZooAttack
from art.estimators.classification import SklearnClassifier
from matplotlib import pyplot as plt

from tree import train_tree, text_label

IMAGE_NAME = path.join('adversarial', 'iot-23.png')


def adversarial_iot():
    colors = ['blue', 'green']
    levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # get the tree model and its training data
    model, x_train, y_train = train_tree(False)
    class_labels = list(set(y_train))
    n_classes = len(class_labels)

    print(x_train)
    print(y_train)

    # Create ART Zeroth Order Optimization attack
    # using scikit-learn DecisionTreeClassifier
    zoo = ZooAttack(
        # A trained classifier
        classifier=SklearnClassifier(model=model),
        # Confidence of adversarial examples: a higher value produces
        # examples that are farther away, from the original input,
        # but classified with higher confidence as the target class.
        confidence=0.25,
        # Should the attack target one specific class
        targeted=False,
        # The initial learning rate for the attack algorithm. Smaller
        # values produce better results but are slower to converge.
        learning_rate=1e-1,
        # The maximum number of iterations.
        max_iter=50,
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
        nb_parallel=1,
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

    # generate plot
    fig, axs = plt.subplots(1, n_classes, figsize=(n_classes * 3, 3))

    for i, class_label in enumerate(class_labels):

        # Plot difference vectors
        for j in range(y_train[y_train == i].shape[0]):
            x_1_0 = x_train[y_train == i][j, 0]
            x_1_1 = x_train[y_train == i][j, 1]
            x_2_0 = x_train_adv[y_train == i][j, 0]
            x_2_1 = x_train_adv[y_train == i][j, 1]
            if x_1_0 != x_2_0 or x_1_1 != x_2_1:
                axs[i].plot([x_1_0, x_2_0], [x_1_1, x_2_1],
                            c='black', zorder=1)

        # Plot benign samples
        for i_class_2 in range(n_classes):
            axs[i].scatter(
                x_train[y_train == i_class_2][:, 0],
                x_train[y_train == i_class_2][:, 1],
                s=20, zorder=2, c=colors[i_class_2])
        axs[i].set_aspect('equal', adjustable='box')

        # Show predicted probability as contour plot
        h = .01
        x_min, x_max = -10, 10
        y_min, y_max = x_min, x_max

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        Z_proba = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        Z_proba = Z_proba[:, i].reshape(xx.shape)
        im = axs[i].contourf(
            xx, yy, Z_proba, levels=levels[:], vmin=0, vmax=1
        )
        if i == n_classes - 1:
            cax = fig.add_axes([0.92, 0.2, 0.01, 0.6])
            plt.colorbar(im, ax=axs[i], cax=cax)

        # Plot adversarial samples
        for j in range(y_train[y_train == i].shape[0]):
            x_1_0 = x_train[y_train == i][j, 0]
            x_1_1 = x_train[y_train == i][j, 1]
            x_2_0 = x_train_adv[y_train == i][j, 0]
            x_2_1 = x_train_adv[y_train == i][j, 1]
            if x_1_0 != x_2_0 or x_1_1 != x_2_1:
                print('adversarial success!')
                axs[i].scatter(
                    x_2_0, x_2_1, zorder=2, c='red', marker='X'
                )

        axs[i].set_xlim((x_min, x_max))
        axs[i].set_ylim((y_min, y_max))
        axs[i].set_title(text_label(i))
        axs[i].set_xlabel('x-axis')
        axs[i].set_ylabel('y-axis')

    plt.savefig(IMAGE_NAME)
    plt.show()


if __name__ == '__main__':
    adversarial_iot()
