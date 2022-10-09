import csv
from collections import Counter
from typing import Optional
from os import path

import numpy as np

from src import BaseUtil, AbsClassifierInstance
from src import Validator, NbTCP, NbUDP, NbOther, IotTCP, IotUDP


class AbsAttack(BaseUtil):

    def __init__(self, name):
        self.name = name
        self.out_dir = None
        self.cls: Optional[AbsClassifierInstance] = None
        self.evasions = np.array([])
        self.adv_x = np.array([])
        self.adv_y = np.array([])
        self.valid_result = np.array([])
        self.validator_kind = None

    @property
    def attack_success(self):
        return len(self.evasions) > 0

    def figure_name(self, n):
        return path.join(self.out_dir, f'{self.name}_{self.cls.name}_{n}.png')

    def set_cls(self, cls: AbsClassifierInstance):
        self.cls = cls
        self.out_dir = cls.out_dir
        return self

    def set_validator(self, kind):
        self.validator_kind = kind
        return self

    def run(self):
        pass

    def validate(self):
        if self.validator_kind is None:
            return
        temp_arr = []
        attrs = self.cls.attrs[:self.cls.n_features]
        for (index, record) in enumerate(self.cls.denormalize(self.adv_x)):
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
            temp_arr.append(not v_inst or Validator.validate(v_inst))
        self.valid_result = np.array(temp_arr)

    def log_attack_setup(self):
        self.show('Attack', self.name)
        self.show('Mutable', ", ".join(self.cls.mutable_attrs))
        self.show('Immutable', ", ".join(self.cls.immutable_attrs))

    def log_attack_stats(self):
        ev, tot = len(self.evasions), self.cls.n_train
        p = 100 * (ev / tot)
        self.show('Total evasions', f'{ev} of {tot} - {p:.1f} %')
        if self.validator_kind:
            v = len([s for s in self.valid_result if s])
            q = 100 * (v / tot)
            self.show('Total valid', f'{v} of {tot} - {q:.1f} %')
            ve = sum([1 for (i, is_valid) in enumerate(self.valid_result)
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
        dump(self.cls.denormalize(self.cls.train_x), self.cls.train_y, 'orid')
        dump(self.adv_x, self.adv_y, 'adv')
        dump(self.cls.denormalize(self.adv_x), self.adv_y, 'advd')
