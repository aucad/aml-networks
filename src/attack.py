import csv
from typing import Optional
from os import path

import numpy as np

from src import BaseUtil, AbsClassifierInstance


class AbsAttack(BaseUtil):

    def __init__(self, name):
        self.name = name
        self.out_dir = None
        self.cls: Optional[AbsClassifierInstance] = None
        self.evasions = np.array([])
        self.adv_x = np.array([])
        self.adv_y = np.array([])

    @property
    def attack_success(self):
        return len(self.evasions) > 0

    def figure_name(self, n):
        return path.join(self.out_dir, f'{self.name}_{self.cls.name}_{n}.png')

    def set_cls(self, cls: AbsClassifierInstance):
        self.cls = cls
        self.out_dir = cls.out_dir
        return self

    def run(self):
        pass

    def log_attack_setup(self):
        self.show('Attack', self.name)
        self.show('Mutable attrs', ", ".join(self.cls.mutable_attrs))
        self.show('Immutable attrs', ", ".join(self.cls.immutable_attrs))

    def log_attack_stats(self):
        ev, tot = len(self.evasions), self.cls.n_train
        p = 100 * (ev / tot)
        self.show('Evasion success', f'{ev} of {tot} - {p:.1f} %')

    def dump_result(self):
        """Write to csv file original and adversarial examples."""

        def dump(x, y, name):
            labels = y[self.evasions].reshape(-1, 1)
            rows = np.append(x[self.evasions, :], labels, 1)
            f_name = path.join(self.out_dir, f'{name}.csv')
            # all masked columns + class label
            int_cols = self.cls.mask_cols + [self.cls.n_features]
            with open(f_name, 'w', newline='') as fp:
                w = csv.writer(fp, delimiter=',')
                w.writerow(self.cls.attrs)
                w.writerows([
                    [int(val) if i in int_cols else val
                     for i, val in enumerate(row)]
                    for row in rows])

        dump(self.cls.train_x, self.cls.train_y, 'ori')
        dump(self.cls.denormalize(self.cls.train_x), self.cls.train_y, 'orid')
        dump(self.adv_x, self.adv_y, 'adv')
        dump(self.cls.denormalize(self.adv_x), self.adv_y, 'advd')
