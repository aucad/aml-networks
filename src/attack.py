from os import path
from typing import Optional

import numpy as np

from src import Classifier, Validator, sdiv, utility


class Attack:

    def __init__(
            self, name, default_iter, validator, uuid, capture, iters
    ):
        self.uuid = uuid
        self.name = name
        self.out_dir = None
        self.plot_result = False
        self.validator_kind = validator
        self.cls: Optional[Classifier] = None
        self.max_iter = default_iter if iters < 1 else iters
        self.save_records = capture
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
    def use_validator(self):
        return self.validator_kind is not None

    @property
    def idx_valid_evades(self):
        if self.use_validator:
            v_ids = [i for i, s in enumerate(self.valid_result) if s]
            both = list(set(v_ids).intersection(set(self.evasions)))
            return np.array(both)
        return self.evasions.copy()

    @property
    def evasion_success(self):
        return sdiv(self.n_evasions, self.n_records) * 100

    @property
    def validation_success(self):
        return sdiv(self.n_valid, self.n_evasions) * 100

    @property
    def has_evasions(self):
        return len(self.idx_valid_evades) > 0

    @property
    def has_invalid(self):
        return len(self.validation_reasons.keys()) > 0

    @property
    def error(self):
        target, x_adv = np.array(self.ori_x), np.array(self.adv_x)
        errors = np.linalg.norm((target - x_adv), axis=1)
        errors = errors[self.evasions] if len(self.evasions) else [0]
        err_min, err_max = 0, 0
        if self.has_evasions:
            err_min = min(errors)
            err_max = max(errors)
        return err_min, err_max

    @property
    def label_stats(self) -> dict:
        result, final_labels = {}, []
        if self.use_validator:
            if self.n_valid > 0:
                final_labels = self.adv_y[self.idx_valid_evades] \
                    .flatten().tolist()
        elif self.n_evasions > 0:
            final_labels = self.adv_y[self.evasions] \
                .flatten().tolist()
        for label in self.cls.classes:
            n = final_labels.count(label)
            key = self.cls.text_label(label)
            result[key] = n
        return result

    def figure_name(self, n):
        return path.join(
            self.out_dir,
            f'{self.uuid}_{self.name}_{self.cls.name}_'
            f'{self.cls.fold_n}_{n}.png')

    def set_cls(self, cls: Classifier):
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

    def eval_examples(self):
        """
        Do prediction on adversarial vs original records and determine
        which records succeed at evading expected class label.
        """
        # predictions for original instances
        ori_in = self.cls.formatter(self.ori_x, self.ori_y)
        original = self.cls.predict(ori_in).flatten().tolist()
        correct = np.array(
            (np.where(np.array(self.ori_y) == original)[0])
            .flatten().tolist())

        # adversarial predictions for same data
        adv_in = self.cls.formatter(self.adv_x, self.ori_y)
        adversarial = self.cls.predict(adv_in).flatten().tolist()
        self.adv_y = np.array(adversarial)

        # adversary succeeds iff predictions differ
        evades = np.array(
            (np.where(self.adv_y != original)[0]).flatten().tolist())
        self.evasions = np.intersect1d(evades, correct)

    def post_run(self):
        self.validate()
        if self.has_evasions:
            self.dump_result()

    def validate(self):
        if self.use_validator and self.n_evasions > 0:
            attrs = self.cls.attrs[:self.cls.n_features]
            records = self.cls.denormalize(self.adv_x)
            result, reasons = Validator.validate_records(
                self.validator_kind, attrs, records)
            self.validation_reasons = reasons
            self.valid_result = np.array(result)

    def dump_result(self):
        """Write to csv file original and adversarial examples."""
        if not self.save_records:
            return
        self.dump(self.ori_x, self.ori_y, 'ori')
        self.dump(self.adv_x, self.adv_y, 'adv')

    def dump(self, x, y, name):
        x = self.cls.denormalize(x.copy())
        attrs = self.cls.attrs[:]
        int_cols = self.cls.mask_cols + [self.cls.n_features]
        labels = y[self.evasions].reshape(-1, 1)
        rows = np.append(x[self.evasions, :], labels, 1)
        fn = f'{self.uuid}_{self.cls.fold_n}_{name}.csv'

        # if validator is used, append validation result
        if self.use_validator and self.n_evasions > 0:
            valid = self.valid_result[self.evasions].reshape(-1, 1)
            int_cols.append(len(rows[0]))
            rows = np.append(rows[:, :], valid, 1)
            attrs.append('valid')

        utility.write_dataset(
            path.join(self.out_dir, fn), attrs, rows, int_cols)
