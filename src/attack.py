import csv
from typing import Optional
from os import path

import numpy as np

from src import BaseUtil, AbsClassifierInstance


class AbsAttack(BaseUtil):

    def __init__(self, name):
        self.name = name
        self.cls: Optional[AbsClassifierInstance] = None
        self.out_dir = None

    def set_cls(self, cls: AbsClassifierInstance):
        self.cls = cls
        self.out_dir = cls.out_dir
        return self

    def run(self):
        pass

    def dump_result(self, evasions, adv_x, adv_y):
        """Write to csv file original and adversarial examples.

        arguments:
            evasions - list of indices where attack succeeded
            adv_x - adversarial examples, np.array (2d)
            adv_y - adversarial labels, np.array (1d)
        """

        def fmt(x, y):
            # append row and label, for each row
            labels = y[evasions].reshape(-1, 1)
            return (np.append(x[evasions, :], labels, 1)).tolist()

        self.ensure_out_dir(self.out_dir)
        inputs = [[fmt(self.cls.train_x, self.cls.train_y), 'ori.csv'],
                  [fmt(adv_x, adv_y), 'adv.csv']]

        for (rows, name) in inputs:
            fname = path.join(self.out_dir, name)
            with open(fname, 'w', newline='') as fp:
                w = csv.writer(fp, delimiter=',')
                w.writerow(self.cls.attrs)
                for row in rows:
                    fmt_row = []
                    for i, val in enumerate(row):
                        int_col = i in self.cls.mask_cols or \
                                  i == len(row) - 1
                        fmt_row.append(int(val) if int_col else val)
                    w.writerow(fmt_row)
