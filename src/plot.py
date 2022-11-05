from itertools import combinations

from matplotlib import pyplot as plt
import numpy as np

from src import AbsAttack, AbsClassifierInstance


class Plotter:

    @staticmethod
    def attack_plot(attack: AbsAttack):
        """Visualize the adversarial attack results"""

        if not attack.plot_result:
            return

        colors = ['deepskyblue', 'lawngreen']
        diff_props = {'c': 'black', 'zorder': 2, 'lw': 1}
        class_props = {'edgecolor': 'black', 'lw': .5, 's': 20,
                       'zorder': 2}
        adv_props = {'zorder': 2, 'c': 'red', 'marker': 'x', 's': 12}
        plt.rc('axes', labelsize=6)
        plt.rc('xtick', labelsize=6)
        plt.rc('ytick', labelsize=6)

        evasions, attr, adv_ex = \
            attack.evasions, attack.cls.attrs, attack.adv_x
        x_train, y_train = attack.cls.train_x, attack.cls.train_y
        class_labels = list([int(i) for i in np.unique(y_train)])

        rows, cols = 2, 3
        subplots = rows * cols
        markers = ['o', 's']
        x_min, y_min, x_max, y_max = -0.1, -0.1, 1.1, 1.1

        non_binary_attrs = attack.non_bin_attributes(x_train)
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
            plt.savefig(attack.figure_name(f + 1))

    @staticmethod
    def classifier_plot(cls: AbsClassifierInstance):
        """Plot classifier instance."""
        cls.tree_plotter()
        plt.tight_layout()
        plt.savefig(cls.plot_path, dpi=200)
