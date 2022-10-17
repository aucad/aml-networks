import csv
from itertools import combinations
from os import path
from typing import Optional

import numpy as np

from src import BaseUtil, AbsClassifierInstance
from src import Validator, \
    NbTCP, NbUDP, NbOther, IotTCP, IotUDP, IotICMP, IotOther
from matplotlib import pyplot as plt


class AbsAttack(BaseUtil):

    def __init__(self, name, iterated):
        self.name = name
        self.out_dir = None
        self.cls: Optional[AbsClassifierInstance] = None
        self.evasions = np.array([])
        self.adv_x = np.array([])
        self.adv_y = np.array([])
        self.valid_result = np.array([])
        self.validator_kind = None
        self.iterated = iterated
        self.max_iter = 100
        self.iter_step = 10

    @property
    def attack_success(self):
        return len(self.evasions) > 0

    def figure_name(self, n):
        return path.join(self.out_dir,
                         f'{self.name}_{self.cls.name}_{n}.png')

    def set_cls(self, cls: AbsClassifierInstance):
        self.cls = cls
        self.out_dir = cls.out_dir
        return self

    @staticmethod
    def non_bin_attributes(np_array):
        """Get column indices of non-binary attributes"""
        return [feat for feat in range(len(np_array[0]))
                if len(list(set(np_array[:, feat]))) > 2]

    def set_validator(self, kind):
        self.validator_kind = kind
        return self

    def run(self, max_iter):
        pass

    def validate(self):
        if self.validator_kind is None:
            return
        temp_arr = []
        attrs = self.cls.attrs[:self.cls.n_features]
        for (index, record) in enumerate(
                self.cls.denormalize(self.adv_x)):
            # make a dictionary of record
            rec_nd = dict([(a, b) for a, b in zip(attrs, record)])
            proto = self.cls.determine_proto(record)
            v_inst = None
            if self.validator_kind == Validator.NB15:
                if proto == self.tcp:
                    v_inst = NbTCP(attrs, **rec_nd)
                elif proto == self.udp:
                    v_inst = NbUDP(attrs, **rec_nd)
                else:
                    v_inst = NbOther(attrs, **rec_nd)
            elif self.validator_kind == Validator.IOT23:
                if proto == self.tcp:
                    v_inst = IotTCP(attrs, **rec_nd)
                elif proto == self.udp:
                    v_inst = IotUDP(attrs, **rec_nd)
                elif proto == self.icmp:
                    v_inst = IotICMP(attrs, **rec_nd)
                else:
                    v_inst = IotOther(attrs, **rec_nd)
            temp_arr.append(not v_inst or Validator.validate(v_inst))
        self.valid_result = np.array(temp_arr)

    def log_attack_setup(self):
        self.show('Attack', self.name)
        self.show('Mutable', ", ".join(self.cls.mutable_attrs))
        self.show('Immutable', ", ".join(self.cls.immutable_attrs))
        self.show('Max iterations', self.max_iter)

    def log_attack_stats(self):
        ev, tot = len(self.evasions), self.cls.n_train
        p = 100 * (ev / tot)
        self.show('Total evasions', f'{ev} of {tot} - {p:.1f} %')
        if self.validator_kind:
            v = len([s for s in self.valid_result if s])
            q = 100 * (v / tot)
            self.show('Total valid', f'{v} of {tot} - {q:.1f} %')
            ve = sum(
                [1 for (i, is_valid) in enumerate(self.valid_result)
                 if is_valid and i in self.evasions])
            r = 100 * (ve / tot)
            self.show('Evades + valid', f'{ve} of {tot} - {r:.1f} %')

    def dump_result(self):
        """Write to csv file original and adversarial examples."""

        def dump(x, y, name):
            attrs = self.cls.attrs[:]
            int_cols = self.cls.mask_cols + [self.cls.n_features]
            labels = y[self.evasions].reshape(-1, 1)
            rows = np.append(x[self.evasions, :], labels, 1)
            if self.validator_kind:
                valid = self.valid_result[self.evasions].reshape(-1, 1)
                int_cols.append(len(rows[0]))
                rows = np.append(rows[:, :], valid, 1)
                attrs.append('valid')

            self.ensure_out_dir(self.out_dir)
            f_name = path.join(self.out_dir, f'{name}.csv')

            # all masked columns + class label
            with open(f_name, 'w', newline='') as fp:
                w = csv.writer(fp, delimiter=',')
                w.writerow(attrs)
                w.writerows([
                    [bool(val) if i == len(self.cls.attrs) + 1 else
                     int(val) if i in int_cols else val
                     for i, val in enumerate(row)]
                    for row in rows])

        dump(self.cls.train_x, self.cls.train_y, 'ori')
        dump(self.cls.denormalize(self.cls.train_x), self.cls.train_y,
             'orid')
        dump(self.adv_x, self.adv_y, 'adv')
        dump(self.cls.denormalize(self.adv_x), self.adv_y, 'advd')

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
