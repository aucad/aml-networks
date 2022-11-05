import csv
from os import path
from typing import Optional

import numpy as np

from src import AbsClassifierInstance, Validator


class AbsAttack:

    def __init__(
            self, name, default_iter, iter_step, iterated, plot,
            validator_kind, uuid, dump_records, max_iter
    ):
        self.uuid = uuid
        self.name = name
        self.iterated = iterated
        self.plot_result = plot
        self.validator_kind = validator_kind
        self.max_iter = default_iter if max_iter < 1 else max_iter
        self.iter_step = iter_step
        self.save_records = dump_records
        self.out_dir = None
        self.cls: Optional[AbsClassifierInstance] = None
        self.evasions = np.array([])
        self.ori_x = np.array([])
        self.ori_y = np.array([])
        self.adv_x = np.array([])
        self.adv_y = np.array([])
        self.valid_result = np.array([])
        self.validation_reasons = {}

    def reset(self):
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
    def n_evasions(self):
        return len(self.evasions)

    @property
    def n_valid(self):
        return len(self.idx_valid_evades)

    @property
    def idx_valid_evades(self):
        if self.validator_kind:
            v_ids = [i for i, s in enumerate(self.valid_result) if s]
            both = list(set(v_ids).intersection(set(self.evasions)))
            return np.array(both)
        return self.evasions.copy()

    @property
    def evasion_success(self):
        if self.n_records > 0:
            return 100 * (self.n_evasions / self.n_records)
        return 0

    @property
    def validation_success(self):
        if self.n_evasions > 0:
            return 100 * (self.n_valid / self.n_evasions)
        return 0

    @property
    def attack_success(self):
        return len(self.evasions) > 0

    @property
    def label_stats(self) -> dict:
        result, final_labels = {}, []
        if self.validator_kind:
            if self.n_valid > 0:
                final_labels = self.adv_y[self.idx_valid_evades] \
                    .flatten().tolist()
        elif self.n_evasions > 0:
            final_labels = self.adv_y[self.evasions] \
                .flatten().tolist()
        for label in self.cls.classes:
            n = final_labels.count(label)
            result[label] = n
        return result

    def printable_label_stats(self):
        counts = sorted(list(self.label_stats.items()),
                        key=lambda x: x[1], reverse=True)
        return [f'{n} * {self.cls.text_label(lbl)}'
                for lbl, n in counts if n > 0]

    def figure_name(self, n):
        return path.join(
            self.out_dir,
            f'{self.uuid}_{self.name}_{self.cls.name}_'
            f'{self.cls.fold_n}_{n}.png')

    def set_cls(self, cls: AbsClassifierInstance):
        self.cls = cls
        self.out_dir = cls.out_dir
        self.ori_x = cls.test_x.copy()
        self.ori_y = cls.test_y.copy()
        return self

    @staticmethod
    def non_bin_attributes(np_array):
        """Get column indices of non-binary attributes"""
        return [feat for feat in range(len(np_array[0]))
                if len(list(set(np_array[:, feat]))) > 2]

    def run(self):
        pass

    def post_run(self):
        self.validate()
        if self.attack_success:
            self.plot()
            self.dump_result()

    def validate(self):
        if self.validator_kind and self.n_evasions > 0:
            attrs = self.cls.attrs[:self.cls.n_features]
            records = self.cls.denormalize(self.adv_x)
            result, reasons = Validator.batch_validate(
                self.validator_kind, attrs, records)
            self.validation_reasons = reasons
            self.valid_result = np.array(result)

    def dump_result(self):
        """Write to csv file original and adversarial examples."""
        if not self.save_records:
            return

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

            f_name = path.join(
                self.out_dir,
                f'{self.uuid}_{self.cls.fold_n}_{name}.csv')

            # all masked columns + class label
            with open(f_name, 'w', newline='') as fp:
                w = csv.writer(fp, delimiter=',')
                w.writerow(attrs)
                w.writerows([
                    [bool(val) if i == len(self.cls.attrs) + 1 else
                     int(val) if i in int_cols else val
                     for i, val in enumerate(row)]
                    for row in rows])

        dump(self.cls.denormalize(self.ori_x), self.ori_y, 'ori')
        dump(self.cls.denormalize(self.adv_x), self.adv_y, 'adv')

    def calculate_error(self):
        target, x_adv = np.array(self.ori_x), np.array(self.adv_x)
        errors = np.linalg.norm((target - x_adv), axis=1)
        errors = errors[self.evasions] if len(self.evasions) else [0]
        err_min, err_max = 0, 0
        if self.attack_success:
            err_min = min(errors)
            err_max = max(errors)
        return err_min, err_max

    @staticmethod
    def clear_one_line():
        cols = 256
        print("\033[A{}\033[A".format(' ' * cols), end='\r')
