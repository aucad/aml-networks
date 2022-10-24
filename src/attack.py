import csv
from itertools import combinations
from os import path
from typing import Optional

import numpy as np

from src import BaseUtil, AbsClassifierInstance
from src import Validator
from matplotlib import pyplot as plt


class AbsAttack(BaseUtil):

    def __init__(self, name, iterated, plot, validator_kind, ds_path, ts):
        self.name = name
        self.iterated = iterated
        self.plot_result = plot
        self.validator_kind = validator_kind
        self.max_iter = 100
        self.iter_step = 10
        self.out_dir = None
        self.cls: Optional[AbsClassifierInstance] = None
        self.evasions = np.array([])
        self.ori_x = np.array([])
        self.ori_y = np.array([])
        self.adv_x = np.array([])
        self.adv_y = np.array([])
        self.valid_result = np.array([])
        self.validation_reasons = {}
        self.ts = ts

        self.show('Attack', self.name)
        self.show('Max iterations', self.max_iter)
        self.validate_dataset(ds_path)

    def reset(self):
        self.out_dir = None
        self.cls = None
        self.evasions = np.array([])
        self.ori_x = np.array([])
        self.ori_y = np.array([])
        self.adv_x = np.array([])
        self.adv_y = np.array([])
        self.valid_result = np.array([])
        self.validation_reasons = {}
        return self

    @property
    def n_records(self):
        return len(self.ori_x)

    @property
    def attack_success(self):
        return len(self.evasions) > 0

    def figure_name(self, n):
        return path.join(
            self.out_dir,
            f'{self.ts}_{self.name}_{self.cls.name}_{self.cls.fold_n}_{n}.png')

    def set_cls(self, cls: AbsClassifierInstance):
        self.cls = cls
        self.out_dir = cls.out_dir
        self.ori_x = cls.test_x.copy()
        self.ori_y = cls.test_y.copy()
        return self

    @staticmethod
    def dump_reasons(reasons):
        count = sum(reasons.values())
        v_reasons = '\n'.join(
            [txt for _, txt in sorted(
                [(v, f'{v} * {k}') for k, v in reasons.items()],
                reverse=True)])
        BaseUtil.show('Invalid total', count)
        if count > 0:
            BaseUtil.show('Invalid reasons', v_reasons)

    @staticmethod
    def non_bin_attributes(np_array):
        """Get column indices of non-binary attributes"""
        return [feat for feat in range(len(np_array[0]))
                if len(list(set(np_array[:, feat]))) > 2]

    def run(self, max_iter):
        pass

    def validate(self):
        if self.validator_kind is not None and len(self.evasions) > 0:
            attrs = self.cls.attrs[:self.cls.n_features]
            records = self.cls.denormalize(self.adv_x)
            result, reasons = Validator.batch_validate(
                self.validator_kind, attrs, records)
            self.validation_reasons = reasons
            self.valid_result = np.array(result)

    def log_attack_setup(self):
        self.show('Max iterations', self.max_iter)

    def validate_dataset(self, ds_path):
        if self.validator_kind:
            indices, reasons = Validator.validate_dataset(
                ds_path, self.validator_kind)
            if 0 < sum(reasons.values()):
                AbsAttack.dump_reasons(reasons)
                # print the indices of invalid records
                rec_i = ' '.join([
                    # offset by 2; for attrs + init index=1
                    str(i + 2) for i, v in enumerate(indices)
                    if not v])
                self.show('Validated', ds_path)
                self.show('records', rec_i)
            else:
                self.show('Validated', f'{ds_path} is valid')

    def log_attack_stats(self):
        ev, tot = len(self.evasions), self.n_records
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
            if v != tot:
                AbsAttack.dump_reasons(self.validation_reasons)
        if ev > 0:
            final_labels = self.adv_y[self.evasions].flatten().tolist()
            evasion_freq = '\n'.join([txt for _, txt in sorted(
                [(final_labels.count(c),
                  f'{final_labels.count(c)} * '
                  f'{self.cls.text_label((c + 1) % 2)} to '
                  f'{self.cls.text_label(c)}')
                 for c in self.cls.classes if final_labels.count(c) > 0]
                , reverse=True)])
            self.show('Evasion classes', evasion_freq)

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
            f_name = path.join(
                self.out_dir, f'{self.ts}_{self.cls.fold_n}_{name}.csv')

            # all masked columns + class label
            with open(f_name, 'w', newline='') as fp:
                w = csv.writer(fp, delimiter=',')
                w.writerow(attrs)
                w.writerows([
                    [bool(val) if i == len(self.cls.attrs) + 1 else
                     int(val) if i in int_cols else val
                     for i, val in enumerate(row)]
                    for row in rows])

        dump(self.ori_x, self.ori_y, 'ori')
        dump(self.cls.denormalize(self.ori_x), self.ori_y, 'ori_denorm')
        dump(self.adv_x, self.adv_y, 'adv')
        dump(self.cls.denormalize(self.adv_x), self.adv_y, 'adv_denorm')

    def plot(self):
        """Visualize the adversarial attack results"""
        if not self.plot_result:
            return

        colors = ['deepskyblue', 'lawngreen']
        diff_props = {'c': 'black', 'zorder': 2, 'lw': 1}
        class_props = {'edgecolor': 'black', 'lw': .5, 's': 20, 'zorder': 2}
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
